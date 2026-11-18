import io
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "type": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "type", "target"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": False,
}


def extract_conversation_facts(messages: list[dict[str, str]]) -> dict:
    """Extract entities and explicitly stated relationships from chat messages."""
    conversation = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role not in {"user", "assistant"} or not content:
            continue
        content = re.split(
            r"(?:^|\n)\s*\*\*Retrieved sources\*\*",
            content,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        if content:
            conversation.append(f"{role.upper()}: {content}")

    if not conversation:
        return {"entities": [], "relationships": []}

    prompt = f"""Extract entities and explicitly stated relationship triples
from this conversation for a conventional knowledge graph.
Use only entities and relationships explicitly stated in the chat. Do not infer
relationships merely because two entities appear together. Treat PDF citations,
file names, and retrieved-source metadata as citation metadata, not entities.
Return only JSON matching this schema:
{json.dumps(FACTS_SCHEMA)}

CONVERSATION:
{chr(10).join(conversation)}"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "format": FACTS_SCHEMA,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    response.raise_for_status()
    result = response.json().get("response")
    if not isinstance(result, str):
        raise ValueError("Ollama returned no conversation graph response.")

    facts = json.loads(result)
    if not isinstance(facts, dict):
        raise ValueError("Ollama returned an invalid conversation graph.")
    if not isinstance(facts.get("entities"), list):
        raise ValueError("Conversation graph entities must be a list.")
    if not isinstance(facts.get("relationships"), list):
        raise ValueError("Conversation graph relationships must be a list.")
    for entity in facts["entities"]:
        if not isinstance(entity, dict) or not isinstance(entity.get("name"), str):
            raise ValueError("Ollama returned an invalid conversation graph entity.")
    for relationship in facts["relationships"]:
        if not isinstance(relationship, dict) or any(
            not isinstance(relationship.get(key), str)
            for key in ("source", "type", "target")
        ):
            raise ValueError(
                "Ollama returned an invalid conversation graph relationship."
            )
    return facts


def build_conversation_graph(facts: dict) -> nx.DiGraph:
    """Build a directed knowledge graph from conversation facts."""
    graph = nx.DiGraph()
    entities: dict[str, str] = {}

    for item in facts["entities"]:
        name = item["name"].strip()
        if name:
            entities.setdefault(name.casefold(), name)

    relationships = []
    seen_relationships = set()
    for item in facts["relationships"]:
        source = item["source"].strip()
        relation = item["type"].strip()
        target = item["target"].strip()
        if not source or not relation or not target:
            continue

        source = entities.setdefault(source.casefold(), source)
        target = entities.setdefault(target.casefold(), target)
        key = (source.casefold(), relation.casefold(), target.casefold())
        if key not in seen_relationships:
            seen_relationships.add(key)
            relationships.append((source, relation, target))

    for name in entities.values():
        graph.add_node(name, label=name, kind="entity")

    for source, relation, target in relationships:
        if graph.has_edge(source, target):
            existing = graph[source][target]["label"].split("\n")
            if relation not in existing:
                existing.append(relation)
                graph[source][target]["label"] = "\n".join(existing)
            graph[source][target]["weight"] += 1
        else:
            graph.add_edge(source, target, label=relation, weight=1)

    return graph


def save_graph(graph: nx.Graph, graphml_path: str, image_path: str):
    """Save a conventional node-and-link knowledge graph."""
    graphml_data = "".join(nx.generate_graphml(graph)).encode("utf-8")
    Path(graphml_path).write_bytes(graphml_data)

    node_count = graph.number_of_nodes()
    fig, ax = plt.subplots(figsize=(max(10, min(18, 8 + node_count * 0.3)), 8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    if node_count == 0:
        ax.text(
            0.5,
            0.5,
            "No entities found in this conversation",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.axis("off")
    else:
        pos = nx.spring_layout(graph, seed=42, k=1.3, iterations=100)
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edge_color="#64748B",
            width=1.5,
            arrows=True,
            arrowsize=18,
            connectionstyle="arc3,rad=0.08",
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_shape="o",
            node_size=[1800 + 300 * graph.degree(node) for node in graph],
            node_color="#93C5FD",
            edgecolors="#1D4ED8",
            linewidths=1.5,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={node: data["label"] for node, data in graph.nodes(data=True)},
            ax=ax,
            font_size=9,
            font_weight="medium",
            font_color="#111827",
        )
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels={
                (source, target): data["label"]
                for source, target, data in graph.edges(data=True)
            },
            ax=ax,
            font_size=8,
            font_color="#374151",
            rotate=False,
            label_pos=0.5,
            bbox={"alpha": 0.8, "color": "white", "pad": 1},
        )
        ax.set_title("Current Conversation Knowledge Graph", fontsize=16, pad=18)
        ax.axis("off")

    fig.tight_layout()
    image_buffer = io.BytesIO()
    fig.savefig(
        image_buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    image_data = image_buffer.getvalue()
    Path(image_path).write_bytes(image_data)
    return image_data, graphml_data
