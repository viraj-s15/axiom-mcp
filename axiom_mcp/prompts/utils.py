"""Utility functions for prompt management."""

import inspect
import types
from collections.abc import Callable
from functools import update_wrapper
from typing import Any, Protocol, TypeGuard, TypeVar, cast, runtime_checkable

from .base import AssistantMessage, Message, Prompt, PromptArgument, UserMessage
from .manager import PromptManager

F = TypeVar("F", bound=Callable[..., Any])


@runtime_checkable
class PromptCallable(Protocol):
    """Protocol for prompt-decorated functions."""

    _prompt: Prompt
    _fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def is_prompt_function(obj: Any) -> TypeGuard[PromptCallable]:
    """Type guard to check if an object is a prompt function."""
    return isinstance(obj, PromptCallable)


class PromptWrapped:
    """Concrete implementation of prompt-decorated functions."""

    _prompt: Prompt
    _fn: Callable[..., Any]

    def __init__(self, fn: Callable[..., Any], prompt_obj: Prompt) -> None:
        self._prompt = prompt_obj
        self._fn = fn
        update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


def prompt(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    version: str = "1.0.0",
    tags: list[str] | None = None,
    manager: PromptManager | None = None,
) -> Callable[[F], F] | F:
    """Decorator to create and optionally register a prompt from a function.

    Examples:
        @prompt(name="greeting", tags=["basic"])
        def greet(name: str) -> str:
            return f"Hello {name}!"

        @prompt(tags=["math"])
        async def calculate(x: float, y: float, operation: str = "add") -> str:
            ops = {
                "add": lambda a, b: a + b,
                "multiply": lambda a, b: a * b
            }
            result = ops[operation](x, y)
            return f"The result of {operation} is {result}"
    """

    def decorator(fn: F) -> F:
        if not callable(fn):
            raise ValueError("Decorator must be applied to a callable")

        prompt_obj = Prompt.from_function(
            fn, name=name, description=description, version=version, tags=tags
        )

        if manager is not None:
            manager.add_prompt(prompt_obj)

        # Create a new subclass for this specific function
        wrapper_cls = types.new_class(
            f"Prompt{fn.__name__.title()}",
            bases=(PromptWrapped,),
            exec_body=lambda ns: ns.update(
                {
                    "__module__": fn.__module__,
                    "__doc__": fn.__doc__,
                }
            ),
        )

        # Create an instance and update its wrapper attributes
        wrapper = wrapper_cls(fn, prompt_obj)

        if inspect.iscoroutinefunction(fn):

            async def async_call(*args: Any, **kwargs: Any) -> Any:
                result = fn(*args, **kwargs)
                if inspect.iscoroutine(result):
                    return await result
                return result

            wrapper.__call__ = async_call.__get__(wrapper, wrapper_cls)

        return cast(F, wrapper)

    if func is None:
        return decorator
    return decorator(func)


def batch_register(manager: PromptManager, *prompts: Prompt | Callable) -> list[Prompt]:
    """Register multiple prompts at once.

    Examples:
        @prompt(name="greet")
        def greet(name: str) -> str:
            return f"Hello {name}!"

        @prompt(name="farewell")
        def farewell(name: str) -> str:
            return f"Goodbye {name}!"

        registered = batch_register(manager, greet, farewell)
    """
    registered = []
    for p in prompts:
        if is_prompt_function(p):
            prompt_obj = p._prompt
        elif isinstance(p, Prompt):
            prompt_obj = p
        else:
            raise ValueError(f"Invalid prompt type: {type(p)}")

        registered.append(manager.add_prompt(prompt_obj))
    return registered


def combine_prompts(
    *funcs: Callable,
    name: str | None = None,
    description: str | None = None,
    version: str = "1.0.0",
    tags: list[str] | None = None,
) -> Prompt:
    """Combine multiple prompt functions into a single prompt.

    The resulting prompt will execute functions based on the following rules:
    1. If any function has an exact match for all its required arguments, run only those functions
    2. If no function has required arguments that match exactly, run all functions with no required args
    """

    async def combined_func(**kwargs: Any) -> list[Message]:
        messages = []
        exact_match_funcs = []
        no_required_funcs = []
        for func in funcs:
            if is_prompt_function(func):
                prompt_obj = func._prompt
                try:
                    valid_args = {
                        k: v
                        for k, v in kwargs.items()
                        if k in [arg.name for arg in prompt_obj.arguments]
                    }

                    required_args = [
                        arg.name for arg in prompt_obj.arguments if arg.required
                    ]

                    if required_args and all(
                        arg in valid_args for arg in required_args
                    ):
                        exact_match_funcs.append((func, valid_args))
                    elif not required_args:
                        no_required_funcs.append((func, valid_args))
                except Exception:
                    continue
            else:
                try:
                    sig = inspect.signature(func)
                    valid_params = {
                        k: v for k, v in kwargs.items() if k in sig.parameters
                    }

                    required_params = {
                        k for k, v in sig.parameters.items() if v.default == v.empty
                    }

                    if required_params and all(
                        p in valid_params for p in required_params
                    ):
                        exact_match_funcs.append((func, valid_params))
                    elif not required_params:
                        no_required_funcs.append((func, valid_params))
                except Exception:
                    continue

        # Second pass - execute functions
        funcs_to_run = exact_match_funcs if exact_match_funcs else no_required_funcs

        for func, args in funcs_to_run:
            try:
                if is_prompt_function(func):
                    result = await func._prompt.render(args)
                else:
                    result = func(**args)
                    if inspect.iscoroutine(result):
                        result = await result

                    if isinstance(result, (str, Message)) or isinstance(result, dict):
                        result = [result]
                    else:
                        result = list(result)

                messages.extend(result)
            except Exception:
                continue

        # Convert any non-Message objects to Messages
        final_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                final_messages.append(msg)
            elif isinstance(msg, str):
                final_messages.append(UserMessage(role="user", content=msg))
            elif isinstance(msg, dict):
                final_messages.append(AssistantMessage(role="assistant", **msg))
            else:
                final_messages.append(UserMessage(role="user", content=str(msg)))

        if not final_messages:
            raise ValueError(
                "No functions could be executed with the provided arguments"
            )

        return final_messages

    arg_map = {}
    func_args = {}

    for func in funcs:
        if is_prompt_function(func):
            prompt_obj = func._prompt
            func_name = prompt_obj.name
            func_args[func_name] = set()

            for arg in prompt_obj.arguments:
                func_args[func_name].add(arg.name)
                if arg.name not in arg_map:
                    arg_map[arg.name] = PromptArgument(
                        name=arg.name,
                        description=arg.description,
                        required=False,
                        type_hint=arg.type_hint,
                        default=arg.default or None,
                    )
        else:
            sig = inspect.signature(func)
            func_name = func.__name__
            func_args[func_name] = set()

            for param_name, param in sig.parameters.items():
                func_args[func_name].add(param_name)
                if param_name not in arg_map:
                    arg_map[param_name] = PromptArgument(
                        name=param_name,
                        description=f"Parameter {param_name} for function {func_name}",
                        required=False,
                        type_hint=(
                            param.annotation.__name__
                            if param.annotation != param.empty
                            else "Any"
                        ),
                        default=None,
                    )

    prompt_name = name or "_".join(f.__name__ for f in funcs)
    return Prompt(
        name=prompt_name,
        description=description,
        version=version,
        arguments=list(arg_map.values()),
        tags=tags or [],
        fn=combined_func,
    )
