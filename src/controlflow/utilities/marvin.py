# import inspect
# from contextlib import contextmanager
# from typing import Any, Callable

# import marvin.ai.text
# from marvin.client.openai import AsyncMarvinClient
# from marvin.settings import temporary_settings as temporary_marvin_settings
# from openai.types.chat import ChatCompletion
# from prefect import task as prefect_task

# from controlflow.utilities.prefect import (
#     create_json_artifact,
# )

# original_classify_async = marvin.classify_async
# original_cast_async = marvin.cast_async
# original_extract_async = marvin.extract_async
# original_generate_async = marvin.generate_async
# original_paint_async = marvin.paint_async
# original_speak_async = marvin.speak_async
# original_transcribe_async = marvin.transcribe_async


# class AsyncControlFlowClient(AsyncMarvinClient):
#     async def generate_chat(self, **kwargs: Any) -> "ChatCompletion":
#         super_method = super().generate_chat

#         @prefect_task(task_run_name="Generate OpenAI chat completion")
#         async def _generate_chat(**kwargs):
#             messages = kwargs.get("messages", [])
#             create_json_artifact(key="prompt", data=messages)
#             response = await super_method(**kwargs)
#             create_json_artifact(key="response", data=response)
#             return response

#         return await _generate_chat(**kwargs)


# def generate_task(name: str, original_fn: Callable):
#     if inspect.iscoroutinefunction(original_fn):

#         @prefect_task(name=name)
#         async def wrapper(*args, **kwargs):
#             create_json_artifact(key="args", data=[args, kwargs])
#             result = await original_fn(*args, **kwargs)
#             create_json_artifact(key="result", data=result)
#             return result
#     else:

#         @prefect_task(name=name)
#         def wrapper(*args, **kwargs):
#             create_json_artifact(key="args", data=[args, kwargs])
#             result = original_fn(*args, **kwargs)
#             create_json_artifact(key="result", data=result)
#             return result

#     return wrapper


# @contextmanager
# def patch_marvin():
#     with temporary_marvin_settings(default_async_client_cls=AsyncControlFlowClient):
#         try:
#             marvin.ai.text.classify_async = generate_task(
#                 "marvin.classify", original_classify_async
#             )
#             marvin.ai.text.cast_async = generate_task(
#                 "marvin.cast", original_cast_async
#             )
#             marvin.ai.text.extract_async = generate_task(
#                 "marvin.extract", original_extract_async
#             )
#             marvin.ai.text.generate_async = generate_task(
#                 "marvin.generate", original_generate_async
#             )
#             marvin.ai.images.paint_async = generate_task(
#                 "marvin.paint", original_paint_async
#             )
#             marvin.ai.audio.speak_async = generate_task(
#                 "marvin.speak", original_speak_async
#             )
#             marvin.ai.audio.transcribe_async = generate_task(
#                 "marvin.transcribe", original_transcribe_async
#             )
#             yield
#         finally:
#             marvin.ai.text.classify_async = original_classify_async
#             marvin.ai.text.cast_async = original_cast_async
#             marvin.ai.text.extract_async = original_extract_async
#             marvin.ai.text.generate_async = original_generate_async
#             marvin.ai.images.paint_async = original_paint_async
#             marvin.ai.audio.speak_async = original_speak_async
#             marvin.ai.audio.transcribe_async = original_transcribe_async
