"""Simplified Streamlit armor penetration calculator."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st


DEFAULT_ARMOR_INTERVAL = 25
DEFAULT_ARMOR_MIN = 75
DEFAULT_ARMOR_MAX = 300
COMPARISON_BASE_ARMOR = 200.0
TABLE_VIEWS = {
    "Efficient HP Multiplier": "ehp_multiplier",
    "Damage Reduction (%)": "damage_reduction",
    "Armor After Penetration": "effective_armor",
}
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def effective_armor(armor: np.ndarray | float, armor_pen_pct: float, lethality: float) -> np.ndarray:
    """Apply percent armor pen first, then flat pen, and floor at 0 armor."""
    armor_array = np.asarray(armor, dtype=float)
    penetrated = armor_array * (1.0 - armor_pen_pct / 100.0) - lethality
    return np.maximum(penetrated, 0.0)


def combine_penetration_percent(*percent_values: float) -> float:
    """Combine multiple armor penetration percentages multiplicatively."""
    remaining_fraction = 1.0
    for value in percent_values:
        remaining_fraction *= 1.0 - value / 100.0
    return (1.0 - remaining_fraction) * 100.0


def damage_reduction_percent(armor: np.ndarray | float) -> np.ndarray:
    """Damage reduction for non-negative armor."""
    armor_array = np.asarray(armor, dtype=float)
    return armor_array / (100.0 + armor_array) * 100.0


def effective_hp_multiplier(armor: np.ndarray | float) -> np.ndarray:
    """Effective HP multiplier, equal to 1 / (1 - r) for damage reduction r."""
    armor_array = np.asarray(armor, dtype=float)
    return 1.0 + armor_array / 100.0


def damage_increase_percent(
    first_effective_armor: np.ndarray | float,
    second_effective_armor: np.ndarray | float,
) -> np.ndarray:
    """Relative damage gain from moving from the first effective armor to the second."""
    first_armor = np.asarray(first_effective_armor, dtype=float)
    second_armor = np.asarray(second_effective_armor, dtype=float)
    first_multiplier = 100.0 / (100.0 + first_armor)
    second_multiplier = 100.0 / (100.0 + second_armor)
    return (second_multiplier / first_multiplier - 1.0) * 100.0


def true_damage_threshold(armor_pen_pct: float, lethality: float) -> float:
    """Return the highest armor value that still becomes 0 effective armor."""
    remaining_fraction = 1.0 - armor_pen_pct / 100.0
    if remaining_fraction <= 0.0:
        return float("inf")
    return lethality / remaining_fraction


def format_pen_expression(expression: str) -> str:
    return re.sub(r"\s+", "", expression.strip())


def format_compact_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_rounded_number(value: float) -> str:
    return f"{value:.0f}"


def format_threshold_value(value: float) -> str:
    if np.isinf(value):
        return "Infinity"
    return format_rounded_number(value)


def safe_eval_math(expression: str) -> float:
    """Evaluate simple arithmetic expressions safely."""
    normalized = expression.strip()
    normalized = normalized.replace("×", "*").replace("÷", "/")
    normalized = normalized.replace("−", "-").replace("–", "-").replace("—", "-")
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
            try:
                return ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
            except ZeroDivisionError as exc:
                raise ValueError("Division by zero is not allowed.") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("Only numbers and + - * / ( ) are allowed.")

    if not normalized:
        raise ValueError("Enter a value.")

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid math expression.") from exc

    return float(_eval(tree))


def parse_pen_expression(expression: str) -> tuple[list[float], float]:
    """Parse plus-separated armor pen values and combine them multiplicatively."""
    cleaned = expression.replace("%", "").strip()
    if not cleaned:
        raise ValueError("Enter at least one armor pen % value.")

    parts = [part.strip() for part in cleaned.split("+")]
    if any(not part for part in parts):
        raise ValueError("Armor pen % must use numbers separated by +, for example 30 + 35.")

    values: list[float] = []
    for part in parts:
        value = safe_eval_math(part)
        if value < 0 or value > 100:
            raise ValueError("Each armor pen % value must be between 0 and 100.")
        values.append(value)

    return values, combine_penetration_percent(*values)


def parse_lethality_expression(expression: str) -> float:
    """Parse arithmetic input for lethality."""
    cleaned = expression.strip()
    if not cleaned:
        raise ValueError("Enter a lethality value.")
    value = safe_eval_math(cleaned)
    if value < 0:
        raise ValueError("Lethality cannot be negative.")
    return value


def init_state() -> None:
    st.session_state.setdefault("rows", [])
    st.session_state.setdefault("next_row_id", 1)
    st.session_state.setdefault("compare_first_id", None)
    st.session_state.setdefault("compare_second_id", None)
    st.session_state.setdefault("selected_metric_label", "Efficient HP Multiplier")


def armor_columns(min_armor: int, max_armor: int, interval: int) -> np.ndarray:
    return np.arange(min_armor, max_armor + 1, interval, dtype=float)


def build_row(pen_expression: str, lethality_expression: str) -> dict[str, Any]:
    pen_parts, combined_pen = parse_pen_expression(pen_expression)
    lethality_total = parse_lethality_expression(lethality_expression)
    row_id = st.session_state.next_row_id
    st.session_state.next_row_id += 1
    return {
        "id": row_id,
        "pen_expression": format_pen_expression(pen_expression),
        "lethality_expression": format_pen_expression(lethality_expression),
        "pen_parts": pen_parts,
        "armor_pen_pct": combined_pen,
        "lethality": lethality_total,
        "label": f"{format_rounded_number(combined_pen)}% + {format_compact_number(lethality_total)}",
    }


def remove_row(row_id: int) -> None:
    st.session_state.rows = [row for row in st.session_state.rows if row["id"] != row_id]
    if st.session_state.compare_first_id == row_id:
        st.session_state.compare_first_id = None
    if st.session_state.compare_second_id == row_id:
        st.session_state.compare_second_id = None


def selected_rows_from_state() -> list[dict[str, Any]]:
    first_id = st.session_state.compare_first_id
    second_id = st.session_state.compare_second_id
    if first_id is None or second_id is None or first_id == second_id:
        return []

    row_lookup = {row["id"]: row for row in st.session_state.rows}
    if first_id not in row_lookup or second_id not in row_lookup:
        return []

    first_row = row_lookup[first_id]
    second_row = row_lookup[second_id]
    first_damage_gain = float(
        damage_increase_percent(
            effective_armor(COMPARISON_BASE_ARMOR, first_row["armor_pen_pct"], first_row["lethality"]),
            effective_armor(COMPARISON_BASE_ARMOR, second_row["armor_pen_pct"], second_row["lethality"]),
        )
    )
    if first_damage_gain < 0:
        return [second_row, first_row]

    return [first_row, second_row]


def metric_values(row: dict[str, Any], armor_values: np.ndarray) -> dict[str, np.ndarray]:
    effective = effective_armor(armor_values, row["armor_pen_pct"], row["lethality"])
    return {
        "effective_armor": effective,
        "damage_reduction": damage_reduction_percent(effective),
        "ehp_multiplier": effective_hp_multiplier(effective),
    }


def format_table_value(value: float, table_key: str) -> str:
    if table_key == "damage_reduction":
        return f"{value:.0f}%"
    if table_key == "ehp_multiplier":
        return f"{value:.2f}x"
    return f"{value:.0f}"


def build_metric_table(
    rows: list[dict[str, Any]],
    armor_values: np.ndarray,
    table_key: str,
    table_label: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    armor_headers = [str(int(value)) for value in armor_values]

    for row in rows:
        values = metric_values(row, armor_values)[table_key]
        record = {table_label: row["label"]}
        for header, value in zip(armor_headers, values):
            record[header] = format_table_value(float(value), table_key)
        records.append(record)

    return pd.DataFrame(records).set_index(table_label)


def build_selected_rows_table(
    first_row: dict[str, Any],
    second_row: dict[str, Any],
    armor_values: np.ndarray,
    table_key: str,
    table_label: str,
) -> pd.DataFrame:
    armor_headers = [str(int(value)) for value in armor_values]
    first_metrics = metric_values(first_row, armor_values)
    second_metrics = metric_values(second_row, armor_values)
    first_values = first_metrics[table_key]
    second_values = second_metrics[table_key]
    delta_values = second_values - first_values
    damage_gain_values = damage_increase_percent(
        first_metrics["effective_armor"],
        second_metrics["effective_armor"],
    )

    records = []
    for row_label, values in (
        (first_row["label"], first_values),
        (second_row["label"], second_values),
        ("Delta", delta_values),
        ("Damage Increase", damage_gain_values),
    ):
        record = {table_label: row_label}
        for header, value in zip(armor_headers, values):
            if row_label == "Damage Increase":
                record[header] = f"{float(value):.1f}%"
            else:
                record[header] = format_table_value(float(value), table_key)
        records.append(record)

    return pd.DataFrame(records).set_index(table_label)


def build_armor_reference_table(armor_values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Reference": "Armor",
                **{str(int(armor_value)): str(int(armor_value)) for armor_value in armor_values},
            },
            {
                "Reference": "Damage Reduction",
                **{
                    str(int(armor_value)): f"{format_rounded_number(float(damage_reduction_percent(armor_value)))}%"
                    for armor_value in armor_values
                },
            },
        ]
    ).set_index("Reference")


def build_combined_plot(
    rows: list[dict[str, Any]],
    armor_values: np.ndarray,
    table_label: str,
    table_key: str,
) -> go.Figure:
    if table_key == "damage_reduction":
        text_template = "%{y:.2f}%"
        y_title = "Damage reduction (%)"
    elif table_key == "ehp_multiplier":
        text_template = "%{y:.2f}x"
        y_title = "Effective HP multiplier"
    else:
        text_template = "%{y:.2f}"
        y_title = "Armor after penetration"

    fig = go.Figure()
    for row in rows:
        row_metrics = metric_values(row, armor_values)
        values = row_metrics[table_key]
        fig.add_trace(
            go.Scatter(
                x=armor_values,
                y=values,
                mode="lines+markers",
                name=row["label"],
                marker={"size": 8},
                line={"width": 3},
                hovertemplate=f"{row['label']}<br>Armor: %{{x:.0f}}<br>{table_label}: {text_template}<extra></extra>",
            )
        )

    for armor_value in armor_values:
        fig.add_vline(
            x=float(armor_value),
            line_width=1,
            line_dash="dot",
            line_color="rgba(15, 118, 110, 0.25)",
        )

    fig.update_layout(
        title=table_label,
        template="plotly_white",
        height=500,
        xaxis={
            "title": "Base armor",
            "tickmode": "array",
            "tickvals": armor_values,
            "ticktext": [str(int(value)) for value in armor_values],
        },
        yaxis={"title": y_title, "rangemode": "tozero"},
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
        legend={"title": "Combo"},
        showlegend=True,
    )
    return fig


def build_damage_increase_plot(
    first_row: dict[str, Any],
    second_row: dict[str, Any],
    armor_values: np.ndarray,
) -> go.Figure:
    first_effective_armor = metric_values(first_row, armor_values)["effective_armor"]
    second_effective_armor = metric_values(second_row, armor_values)["effective_armor"]
    damage_gain = damage_increase_percent(first_effective_armor, second_effective_armor)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=armor_values,
            y=damage_gain,
            mode="lines+markers",
            name=f"{second_row['label']} vs {first_row['label']}",
            marker={"size": 8},
            line={"width": 3},
            hovertemplate=(
                f"{second_row['label']} vs {first_row['label']}"
                "<br>Armor: %{x:.0f}<br>Damage increase: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    for armor_value in armor_values:
        fig.add_vline(
            x=float(armor_value),
            line_width=1,
            line_dash="dot",
            line_color="rgba(15, 118, 110, 0.25)",
        )

    fig.update_layout(
        title="Damage Increase From Added Armor Pen",
        template="plotly_white",
        height=500,
        xaxis={
            "title": "Base armor",
            "tickmode": "array",
            "tickvals": armor_values,
            "ticktext": [str(int(value)) for value in armor_values],
        },
        yaxis={"title": "Damage increase (%)"},
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
        showlegend=False,
    )
    return fig


def render_sidebar_controls() -> np.ndarray:
    with st.sidebar:
        st.subheader("Armor Columns")
        min_armor, max_armor = st.slider(
            "Armor range",
            min_value=0,
            max_value=500,
            value=(DEFAULT_ARMOR_MIN, DEFAULT_ARMOR_MAX),
            step=5,
        )
        armor_interval = int(
            st.slider(
                "Armor interval",
                min_value=5,
                max_value=100,
                value=DEFAULT_ARMOR_INTERVAL,
                step=5,
            )
        )
        armor_values = armor_columns(int(min_armor), int(max_armor), armor_interval)
        st.caption("Columns use the selected minimum armor, maximum armor, and interval.")

        st.divider()
        st.subheader("Add Row")
        with st.form("add_row_form", clear_on_submit=False):
            pen_expression = st.text_input(
                "Armor pen %",
                value="30 + 35",
                help="Use + between pen sources. Example: 30 + 35 becomes 54.50% total pen.",
            )
            lethality_expression = st.text_input(
                "Lethality",
                value="2*18 + 15",
                help="Arithmetic is allowed. Example: 18 + 20 + 10 becomes 48.",
            )
            submitted = st.form_submit_button("Add Row", use_container_width=True)

        if submitted:
            try:
                st.session_state.rows.append(build_row(pen_expression, lethality_expression))
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        st.divider()
        rows = st.session_state.rows
        if not rows:
            st.subheader("Rows")
            st.caption("No rows added yet.")
        else:
            row_ids = [row["id"] for row in rows]
            label_lookup = {row["id"]: row["label"] for row in rows}

            st.subheader("Compare Rows")
            st.selectbox(
                "Row A",
                options=[None, *row_ids],
                key="compare_first_id",
                format_func=lambda row_id: "None" if row_id is None else label_lookup[row_id],
            )
            st.selectbox(
                "Row B",
                options=[None, *row_ids],
                key="compare_second_id",
                format_func=lambda row_id: "None" if row_id is None else label_lookup[row_id],
            )

            st.divider()
            st.subheader("Delete Row")
            delete_row_id = st.selectbox(
                "Remove",
                options=[None, *row_ids],
                format_func=lambda row_id: "None" if row_id is None else label_lookup[row_id],
            )
            if st.button("Delete Selected Row", use_container_width=True):
                if delete_row_id is not None:
                    remove_row(delete_row_id)
                    st.rerun()

            st.divider()
            st.subheader("Rows")
            for row in rows:
                st.caption(row["label"])

        if st.button("Clear Rows", use_container_width=True):
            st.session_state.rows = []
            st.session_state.compare_first_id = None
            st.session_state.compare_second_id = None
            st.rerun()

    return armor_values


def render_saved_rows() -> None:
    rows = st.session_state.rows
    if not rows:
        st.info("Add at least one row to see the saved row summary.")
        return

    summary = pd.DataFrame(
        [
            {
                "Row": row["label"],
                "Armor pen": row["pen_expression"],
                "Total armor pen": f"{format_rounded_number(row['armor_pen_pct'])}%",
                "Lethality": format_compact_number(row["lethality"]),
                "0 effective armor threshold": format_threshold_value(
                    true_damage_threshold(row["armor_pen_pct"], row["lethality"])
                ),
            }
            for row in rows
        ]
    )
    st.dataframe(summary.set_index("Row"), use_container_width=True)


def render_table_section(
    armor_values: np.ndarray,
    selected_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    st.subheader("All Rows")
    table_label = st.radio(
        "Metric",
        list(TABLE_VIEWS.keys()),
        key="selected_metric_label",
        horizontal=True,
    )
    table_key = TABLE_VIEWS[table_label]

    if not st.session_state.rows:
        st.info("Add at least one row to generate the table.")
        return table_label, table_key

    table = build_metric_table(st.session_state.rows, armor_values, table_key, table_label)
    st.dataframe(table, use_container_width=True)

    if len(selected_rows) == 2:
        st.subheader("Comparison")
        st.dataframe(
            build_selected_rows_table(selected_rows[0], selected_rows[1], armor_values, table_key, table_label),
            use_container_width=True,
        )

    return table_label, table_key


def render_graphs(armor_values: np.ndarray, table_label: str, table_key: str, selected_rows: list[dict[str, Any]]) -> None:
    rows_to_plot = selected_rows if len(selected_rows) == 2 else st.session_state.rows
    if not rows_to_plot:
        return

    st.subheader("Graph")
    st.plotly_chart(
        build_combined_plot(rows_to_plot, armor_values, table_label, table_key),
        use_container_width=True,
    )
    if len(selected_rows) == 2:
        st.subheader("Damage Increase")
        st.plotly_chart(
            build_damage_increase_plot(selected_rows[0], selected_rows[1], armor_values),
            use_container_width=True,
        )
    st.subheader("Armor Reference")
    st.dataframe(build_armor_reference_table(armor_values), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Armor Pen Table Calculator", layout="wide")
    init_state()

    st.title("Armor Pen Table Calculator")
    st.caption(
        "Build rows from armor pen and lethality combos, then switch the table between armor after penetration and damage reduction."
    )

    armor_values = render_sidebar_controls()
    selected_rows = selected_rows_from_state()

    left_col, right_col = st.columns([1, 1.5], gap="large")

    with left_col:
        st.subheader("Saved Rows")
        render_saved_rows()

    with right_col:
        table_label, table_key = render_table_section(armor_values, selected_rows)

    render_graphs(armor_values, table_label, table_key, selected_rows)


if __name__ == "__main__":
    main()
