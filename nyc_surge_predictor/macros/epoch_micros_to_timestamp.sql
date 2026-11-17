{#
  The raw trip timestamps arrive as integers: microseconds since the Unix
  epoch (Spark writes them that way and Snowflake's INFER_SCHEMA keeps them
  numeric). Each warehouse converts them differently, so the conversion is
  dispatched per adapter. This is the only warehouse-specific SQL in the project.
#}
{% macro epoch_micros_to_timestamp(column) -%}
    {{ return(adapter.dispatch('epoch_micros_to_timestamp')(column)) }}
{%- endmacro %}

{% macro default__epoch_micros_to_timestamp(column) -%}
    to_timestamp_ntz({{ column }}, 6)
{%- endmacro %}

{% macro duckdb__epoch_micros_to_timestamp(column) -%}
    make_timestamp({{ column }})
{%- endmacro %}
