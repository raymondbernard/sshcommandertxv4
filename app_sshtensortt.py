import os
import time
import json
import logging
import gc
import torch
from pathlib import Path
from trt_llama_api import TrtLlmAPI

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from collections import defaultdict
from llama_index import ServiceContext
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import set_global_service_context
from faiss_vector_storage import FaissEmbeddingStorage
from ui.user_interface import MainInterface
from app_process_data import Process_Data 
# Setup logging
logging.basicConfig(level=logging.INFO)

app_config_file = 'config\\app_config.json'
model_config_file = 'config\\config.json'
preference_config_file = 'config\\preferences.json'
data_source = 'directory'
# Added By Ray Bernard
# call the app_process_data after chat_logs have been created
def process_sshcmd():
    Process_Data()
    print("Processed Data into SSH Commander DB")

# Where we create the chatl_log
def log_response(query, response, session_id):
    print(response)

    log_entry = {
        "session_id": session_id,
        "query": query,
        "response": response,
        "timestamp": time.time()
    }
    with open("chat_logs.jsonl", "a") as log_file:
        json.dump(log_entry, log_file)
        log_file.write("\n")  # For readability in the log file
    process_sshcmd()

def log_completion_response(completion_response):
    print(f"Received input: {completion_response}, Type: {type(completion_response)}")  # Debug print
    try:
        # Assuming CompletionResponse has a .text attribute
        if not hasattr(completion_response, 'text'):
            raise ValueError("completion_response must have a 'text' attribute.")
        
        user_prompt_text = getattr(completion_response, 'text', None)
        if user_prompt_text is None:
            raise ValueError("The 'text' attribute could not be found.")
        
        log_entry = {
            "user_prompt": user_prompt_text,
            "timestamp": time.time()
        }
        with open("test_user_prompt.jsonl", "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")
    except Exception as e:
        print(f"Failed to log completion response: {e}")


def read_config(file_name):
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
    except json.JSONDecodeError:
        print(f"There was an error decoding the JSON from the file {file_name}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def get_model_config(config, model_name=None):
    models = config["models"]["supported"]
    selected_model = next((model for model in models if model["name"] == model_name), models[0])
    return {
        "model_path": os.path.join(os.getcwd(), selected_model["metadata"]["model_path"]),
        "engine": selected_model["metadata"]["engine"],
        "tokenizer_path": os.path.join(os.getcwd(), selected_model["metadata"]["tokenizer_path"]),
        "max_new_tokens": selected_model["metadata"]["max_new_tokens"],
        "max_input_token": selected_model["metadata"]["max_input_token"],
        "temperature": selected_model["metadata"]["temperature"]
    }


def get_data_path(config):
    return os.path.join(os.getcwd(), config["dataset"]["path"])

# read the app specific config
app_config = read_config(app_config_file)
streaming = app_config["streaming"]
similarity_top_k = app_config["similarity_top_k"]
is_chat_engine = app_config["is_chat_engine"]
embedded_model_name = app_config["embedded_model"]
embedded_model = os.path.join(os.getcwd(), "model", embedded_model_name)
embedded_dimension = app_config["embedded_dimension"]

# read model specific config
selected_model_name = None
selected_data_directory = None
config = read_config(model_config_file)
if os.path.exists(preference_config_file):
    perf_config = read_config(preference_config_file)
    selected_model_name = perf_config.get('models', {}).get('selected')
    selected_data_directory = perf_config.get('dataset', {}).get('path')

if selected_model_name == None:
    selected_model_name = config["models"].get("selected")

model_config = get_model_config(config, selected_model_name)
trt_engine_path = model_config["model_path"]
trt_engine_name = model_config["engine"]
tokenizer_dir_path = model_config["tokenizer_path"]
data_dir = config["dataset"]["path"] if selected_data_directory == None else selected_data_directory

# create trt_llm engine object
llm = TrtLlmAPI(
    model_path=model_config["model_path"],
    engine_name=model_config["engine"],
    tokenizer_dir=model_config["tokenizer_path"],
    temperature=model_config["temperature"],
    max_new_tokens=model_config["max_new_tokens"],
    context_window=model_config["max_input_token"],
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False
)

# create embeddings model object
embed_model = HuggingFaceEmbeddings(model_name=embedded_model)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model,
                                               context_window=model_config["max_input_token"], chunk_size=512,
                                               chunk_overlap=200)
set_global_service_context(service_context)


def generate_inferance_engine(data, force_rewrite=False):
    """
       Initialize and return a FAISS-based inference engine.

       Args:
           data: The directory where the data for the inference engine is located.
           force_rewrite (bool): If True, force rewriting the index.

       Returns:
           The initialized inference engine.

       Raises:
           RuntimeError: If unable to generate the inference engine.
       """
    try:
        global engine
        faiss_storage = FaissEmbeddingStorage(data_dir=data,
                                              dimension=embedded_dimension)
        faiss_storage.initialize_index(force_rewrite=force_rewrite)
        engine = faiss_storage.get_engine(is_chat_engine=is_chat_engine, streaming=streaming,
                                          similarity_top_k=similarity_top_k)
    except Exception as e:
        raise RuntimeError(f"Unable to generate the inference engine: {e}")


# load the vectorstore index
generate_inferance_engine(data_dir)


def stream_chatbot(query, chat_history, session_id):
    print("439 chat history = ", chat_history)
    if data_source == "nodataset":
        for response in call_llm_streamed(query):
            yield response
        return

    if is_chat_engine:
        response = engine.stream_chat(query)
    else:
        response = engine.query(query)

    partial_response = ""
    if len(response.source_nodes) == 0:
        response = llm.stream_complete(query)
        for token in response:
            partial_response += token.delta
            yield partial_response
    else:
        # Aggregate scores by file
        file_scores = defaultdict(float)
        for node in response.source_nodes:
            if 'filename' in node.metadata:
                file_name = node.metadata['filename']
                file_scores[file_name] += node.score

        # Find the file with the highest aggregated score
        highest_score_file = max(file_scores, key=file_scores.get, default=None)

        file_links = []
        seen_files = set()
        for token in response.response_gen:
            partial_response += token
            yield partial_response
            time.sleep(0.05)

        time.sleep(0.2)

        if highest_score_file:
            abs_path = Path(os.path.join(os.getcwd(), highest_score_file.replace('\\', '/')))
            file_name = os.path.basename(abs_path)
            file_name_without_ext = abs_path.stem
            if file_name not in seen_files:  # Check if file_name is already seen
                if data_source == 'directory':
                    file_link = file_name
                else:
                    exit("Wrong data_source type")
                file_links.append(file_link)
                seen_files.add(file_name)  # Add file_name to the set

        if file_links:
            partial_response += "<br>Reference files:<br>" + "<br>".join(file_links)
        yield partial_response

    # call garbage collector after inference
    torch.cuda.empty_cache()
    gc.collect()

interface = MainInterface(chatbot=stream_chatbot if streaming else chatbot, streaming=streaming)

def on_shutdown_handler(session_id):
    global llm, service_context, embed_model, faiss_storage, engine
    import gc
    if llm is not None:
        llm.unload_model()
        del llm
    # Force a garbage collection cycle
    gc.collect()


interface.on_shutdown(on_shutdown_handler)


def reset_chat_handler(session_id):
    global faiss_storage
    global engine
    print('reset chat called', session_id)
    if is_chat_engine == True:
        faiss_storage.reset_engine(engine)


interface.on_reset_chat(reset_chat_handler)


def on_dataset_path_updated_handler(source, new_directory, video_count, session_id):
    print('data set path updated to ', source, new_directory, video_count, session_id)
    global engine
    global data_dir
    if source == 'directory':
        if data_dir != new_directory:
            data_dir = new_directory
            generate_inferance_engine(data_dir)

interface.on_dataset_path_updated(on_dataset_path_updated_handler)


def on_model_change_handler(model, metadata, session_id):
    model_path = os.path.join(os.getcwd(), metadata.get('model_path', None))
    engine_name = metadata.get('engine', None)
    tokenizer_path = os.path.join(os.getcwd(), metadata.get('tokenizer_path', None))

    if not model_path or not engine_name:
        print("Model path or engine not provided in metadata")
        return

    global llm, embedded_model, engine, data_dir, service_context

    if llm is not None:
        llm.unload_model()
        del llm

    llm = TrtLlmAPI(
        model_path=model_path,
        engine_name=engine_name,
        tokenizer_dir=tokenizer_path,
        temperature=metadata.get('temperature', 0.1),
        max_new_tokens=metadata.get('max_new_tokens', 512),
        context_window=metadata.get('max_input_token', 512),
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False
    )
    service_context = ServiceContext.from_service_context(service_context=service_context, llm=llm)
    set_global_service_context(service_context)
    generate_inferance_engine(data_dir)


interface.on_model_change(on_model_change_handler)


def on_dataset_source_change_handler(source, path, session_id):

    global data_source, data_dir, engine
    data_source = source

    if data_source == "nodataset":
        print(' No dataset source selected', session_id)
        return
    
    print('dataset source updated ', source, path, session_id)
    
    if data_source == "directory":
        data_dir = path
    else:
        print("Wrong data type selected")
    generate_inferance_engine(data_dir)

interface.on_dataset_source_updated(on_dataset_source_change_handler)

def handle_regenerate_index(source, path, session_id):
    generate_inferance_engine(path, force_rewrite=True)
    print("on regenerate index", source, path, session_id)

interface.on_regenerate_index(handle_regenerate_index)
# render the interface
interface.render()
