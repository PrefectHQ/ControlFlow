from marvin.beta.assistants.threads import Message, Thread

THREAD_REGISTRY = {}


def save_thread(name: str, thread: Thread):
    """
    Save an OpenAI thread to the thread registry under a known name
    """
    THREAD_REGISTRY[name] = thread


def load_thread(name: str):
    """
    Load an OpenAI thread from the thread registry by name
    """
    if name not in THREAD_REGISTRY:
        thread = Thread()
        save_thread(name, thread)
    return THREAD_REGISTRY[name]


def get_history(thread_id: str, limit: int = None) -> list[Message]:
    """
    Get the history of a thread
    """
    return Thread(id=thread_id).get_messages(limit=limit)
