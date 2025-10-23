"""Utility functions for handling API errors in the UI."""

import logging
from typing import Callable, Optional

from nicegui import ui

logger = logging.getLogger(__name__)


def show_api_error(
    container,
    error: Exception,
    title: str = "Error",
    message: str = "An error occurred",
    retry_callback: Optional[Callable] = None,
):
    """Display a user-friendly error message in the UI.

    Args:
        container: The UI container to render the error in
        error: The exception that was raised
        title: Title for the error card
        message: User-friendly message explaining the error
        retry_callback: Optional callback function for retry button
    """
    container.clear()
    with container:
        with ui.card().classes("fantasy-panel border-2 border-red-500"):
            ui.label(f"⚠️ {title}").classes("text-h5 text-red-400 mb-4")
            ui.label(message).classes("text-gray-300 mb-2")
            ui.label(f"Error: {str(error)}").classes(
                "text-sm text-red-300 mb-4 font-mono"
            )
            ui.label(
                "This may be due to API overload or connectivity issues. "
                "Please try again in a moment."
            ).classes("text-gray-400 text-sm mb-4")

            if retry_callback:
                ui.button("🔄 Try Again", on_click=retry_callback).classes("mt-2")


def show_loading(
    container,
    title: str = "Loading...",
    message: str = "Please wait...",
):
    """Display a loading indicator in the UI.

    Args:
        container: The UI container to render the loading indicator in
        title: Title for the loading card
        message: Message to display while loading
    """
    container.clear()
    with container:
        with ui.card().classes("fantasy-panel"):
            ui.label(title).classes("text-h5 mb-4")
            ui.spinner(size="lg")
            ui.label(message).classes("loading-message mt-4")


async def with_loading_and_error_handling(
    container,
    async_func: Callable,
    loading_title: str = "Loading...",
    loading_message: str = "Please wait...",
    error_title: str = "Error",
    error_message: str = "An error occurred",
    retry_func: Optional[Callable] = None,
):
    """Execute an async function with loading indicator and error handling.

    Args:
        container: The UI container to render loading/errors in
        async_func: The async function to execute
        loading_title: Title for the loading indicator
        loading_message: Message for the loading indicator
        error_title: Title for error messages
        error_message: User-friendly error message
        retry_func: Optional function to retry on error

    Returns:
        The result of async_func, or None if an error occurred
    """
    show_loading(container, loading_title, loading_message)

    try:
        result = await async_func()
        return result
    except Exception as e:
        logger.error(f"{error_title}: {e}", exc_info=True)
        show_api_error(
            container,
            error=e,
            title=error_title,
            message=error_message,
            retry_callback=retry_func,
        )
        return None
