"""
Utilities package initialization.
Provides imports for all utility modules.
"""

from services.utils.code_utils import (
    detect_language,
    find_cursor_position,
    split_at_cursor,
    parse_code_structure
)

from services.utils.file_utils import (
    read_file,
    write_file,
    create_file_hash,
    get_file_size
)

from services.utils.token_utils import (
    count_tokens,
    estimate_tokens,
    is_within_context_limit
)

from services.utils.prompt_utils import (
    optimize_prompt_for_completion,
    format_chat_prompt,
    format_fim_prompt
)