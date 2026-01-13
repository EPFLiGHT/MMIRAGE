import re
from mirage.config.variables import ProcessingParams

from typing import Any, Set

def validate_processing_params(params: ProcessingParams) -> None:
    """
    Validate that ProcessingParams.output_schema uses only variables defined in
    inputs and outputs.

    Raises:
        ValueError: If undefined variables are found in output_schema or prompts
    """

    # Collect all defined variable names
    defined_vars = set()

    # From inputs
    for input_var in params.inputs:
        defined_vars.add(input_var.name)

    # From outputs
    for variable in params.outputs:
        defined_vars.add(variable.name)

    # Extract all template variables (patterns like {var_name})
    def extract_template_vars(obj: Any) -> Set[str]:
        """Recursively extract all {var} patterns from templates."""
        vars_found = set()

        if isinstance(obj, str):
            # Find all {variable_name} patterns
            matches = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", obj)
            vars_found.update(matches)
        elif isinstance(obj, dict):
            for value in obj.values():
                vars_found.update(extract_template_vars(value))
        elif isinstance(obj, list):
            for item in obj:
                vars_found.update(extract_template_vars(item))

        return vars_found

    # Check output_schema templates
    schema_vars = extract_template_vars(params.output_schema)

    # Check output prompts
    prompt_vars = set()
    for variable in params.outputs:
        prompt_vars.update(extract_template_vars(variable.prompt))

    # Validate all template variables are defined
    undefined_in_schema = schema_vars - defined_vars
    undefined_in_prompts = prompt_vars - defined_vars

    errors = []

    if undefined_in_schema:
        errors.append(
            f"Undefined variables in output_schema: {', '.join(sorted(undefined_in_schema))}. "
            f"Available variables: {', '.join(sorted(defined_vars))}"
        )

    if undefined_in_prompts:
        errors.append(
            f"Undefined variables in output prompts: {', '.join(sorted(undefined_in_prompts))}. "
            f"Available variables: {', '.join(sorted(defined_vars))}"
        )

    if errors:
        raise ValueError("\n".join(errors))

    print(
        f"✅ ProcessingParams validation passed. "
        f"Defined variables: {', '.join(sorted(defined_vars))}"
    )


