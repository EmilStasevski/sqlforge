{# Use the configured schema name verbatim (main / staging) instead of <target>_<custom>. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
  {{ custom_schema_name if custom_schema_name is not none else target.schema }}
{%- endmacro %}
