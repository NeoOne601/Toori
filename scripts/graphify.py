import sqlite3
import os
import json
import hashlib
from collections import defaultdict

DB_PATH = ".code-review-graph/graph.db"
OUT_DIR = "graphify-out"
CACHE_DIR = os.path.join(OUT_DIR, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "state.json")

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_db_hash(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), MAX(updated_at) FROM nodes")
    node_stat = cursor.fetchone()
    cursor.execute("SELECT COUNT(*), MAX(updated_at) FROM edges")
    edge_stat = cursor.fetchone()
    state_str = f"N:{node_stat[0]}|{node_stat[1]}_E:{edge_stat[0]}|{edge_stat[1]}"
    return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found. Please run mcp_code-review-graph_build_or_update_graph_tool first.")
        return

    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    
    current_hash = get_db_hash(conn)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            if cache.get("db_hash") == current_hash:
                print("Graph unchanged. Skipping regeneration.")
                return

    print("Generating graphify output...")
    cursor = conn.cursor()

    # Export Nodes, Edges, Communities
    cursor.execute("SELECT id, name FROM communities")
    communities = {r[0]: r[1] for r in cursor.fetchall()}

    cursor.execute("SELECT qualified_name, name, kind, community_id, file_path FROM nodes")
    raw_nodes = cursor.fetchall()
    
    vis_nodes = []
    node_communities = {}
    for r in raw_nodes:
        qn, name, kind, comm_id, file_path = r
        node_communities[qn] = comm_id
        vis_nodes.append({
            "id": qn,
            "label": name,
            "title": f"[{kind}] {qn}\nFile: {file_path}",
            "group": communities.get(comm_id, "Unknown"),
            "community_id": comm_id,
            "value": 1  # Base metric for size, will be updated
        })

    cursor.execute("SELECT source_qualified, target_qualified, kind FROM edges")
    raw_edges = cursor.fetchall()
    
    vis_edges = []
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    inter_community_edges = []
    
    # Precompute valid node ids to prevent vis.js errors on missing targets
    valid_nodes = set(n["id"] for n in vis_nodes)

    for r in raw_edges:
        src, tgt, kind = r
        if src not in valid_nodes or tgt not in valid_nodes:
            continue
            
        vis_edges.append({
            "from": src,
            "to": tgt,
            "title": kind,
            "arrows": "to"
        })
        in_degree[tgt] += 1
        out_degree[src] += 1
        
        c_src = node_communities.get(src)
        c_tgt = node_communities.get(tgt)
        if c_src and c_tgt and c_src != c_tgt:
            inter_community_edges.append((src, tgt, kind, c_src, c_tgt))

    # Scale nodes by degree
    for n in vis_nodes:
        deg = in_degree.get(n["id"], 0) + out_degree.get(n["id"], 0)
        n["value"] += deg

    graph_data = {"nodes": vis_nodes, "edges": vis_edges}
    
    # Dump graph.json
    with open(os.path.join(OUT_DIR, "graph.json"), "w") as f:
        json.dump(graph_data, f)
        
    # Generate GRAPH_REPORT.md
    top_in = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    top_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    
    report = ["# Codebase Graph Report\n"]
    report.append("## God Nodes")
    report.append("Nodes that are heavily depended upon (High In-Degree):")
    for qn, count in top_in:
        report.append(f"- **`{qn}`** ({count} incoming edges)")
        
    report.append("\nNodes that coordinate many others (High Out-Degree):")
    for qn, count in top_out:
        report.append(f"- **`{qn}`** ({count} outgoing edges)")
        
    report.append("\n## Surprising Connections")
    report.append("Edges bridging distant code communities:")
    # Group inter-community edges by frequency to find top surprising connections
    bridge_counts = defaultdict(int)
    for src, tgt, kind, cs, ct in inter_community_edges:
        bridge_counts[(src, tgt)] += 1
        
    top_bridges = sorted(bridge_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    for (src, tgt), _ in top_bridges:
        src_c = communities.get(node_communities.get(src))
        tgt_c = communities.get(node_communities.get(tgt))
        report.append(f"- **`{src}`** *(Community: {src_c})* \u2192 **`{tgt}`** *(Community: {tgt_c})*")
        
    report.append("\n## Suggested Questions to Explore")
    for idx, ((src, tgt), _) in enumerate(top_bridges[:3]):
        report.append(f"{idx+1}. Why does `{src}` depend heavily on `{tgt}` despite them belonging to completely different domains?")
        
    with open(os.path.join(OUT_DIR, "GRAPH_REPORT.md"), "w") as f:
        f.write("\n".join(report))
        
    # Generate graph.html embedding graph_data natively
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Code Review Graph Visualization</title>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #mynetwork {{ width: 100vw; height: 100vh; border: 1px solid lightgray; }}
        body {{ margin: 0; padding: 0; font-family: sans-serif; overflow: hidden; }}
        #hud {{ position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; z-index: 100; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        #debug {{ position: absolute; bottom: 10px; left: 10px; color: red; background: yellow; z-index: 999; }}
    </style>
</head>
<body>
<div id="hud">
    <h3>Codebase Graph</h3>
    <input type="text" id="search" placeholder="Search Node..." onkeyup="searchNodes()" style="padding: 5px; width: 250px;">
</div>
<div id="debug"></div>
<div id="mynetwork"></div>

<script type="text/javascript">
    window.onerror = function(msg, url, lineNo, columnNo, error) {{
        document.getElementById('debug').innerText += "ERROR: " + msg + "\\n";
        return false;
    }};
    console.log("Starting Graphify setup...");
    document.getElementById('debug').innerText = "Loading data...";
    
    // Defer graph initialization so UI renders first
    setTimeout(function() {{
        try {{
            document.getElementById('debug').innerText = "Parsing JSON data (" + {len(json.dumps(graph_data))} + " bytes)...";
            const graphData = {json.dumps(graph_data)};
            console.log("Data loaded. Nodes:", graphData.nodes.length, "Edges:", graphData.edges.length);
            document.getElementById('debug').innerText = "Initializing Vis.js network for " + graphData.nodes.length + " nodes...";
            
            const nodes = new vis.DataSet(graphData.nodes);
            const edges = new vis.DataSet(graphData.edges);
        
            const container = document.getElementById('mynetwork');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
        nodes: {{
            shape: 'dot',
            scaling: {{
                min: 10,
                max: 50,
                label: {{ min: 8, max: 20, drawThreshold: 12, maxVisible: 20 }}
            }},
            font: {{ size: 12, face: 'Tahoma' }}
        }},
        edges: {{
            width: 0.15,
            color: {{ inherit: 'from' }},
            smooth: false
        }},
        physics: {{
            barnesHut: {{ gravitationalConstant: -2000, centralGravity: 0.3, springLength: 95, springConstant: 0.04, damping: 0.09 }},
            stabilization: {{ iterations: 50 }}
        }},
        layout: {{
            improvedLayout: false
        }}
    }};
    
    // Check if graph is very large and adjust physics to avoid hanging browser
    if (graphData.nodes.length > 1000) {{
        options.physics.stabilization.iterations = 15; 
        options.physics.barnesHut.gravitationalConstant = -1000;
        options.interaction = {{ hideEdgesOnDrag: true }};
    }}

    const network = new vis.Network(container, data, options);

    network.on("stabilizationIterationsDone", function () {{
        network.setOptions({{ physics: false }});
    }});

    window.searchNodes = function() {{
        const query = document.getElementById('search').value.toLowerCase();
        if (!query) return;
        const matching = graphData.nodes.filter(n => n.label.toLowerCase().includes(query) || n.id.toLowerCase().includes(query));
        if (matching.length > 0) {{
            network.focus(matching[0].id, {{ scale: 1.5, animation: true }});
            network.selectNodes([matching[0].id]);
        }}
    }}
        }} catch(e) {{
            document.getElementById('debug').innerText += "Exception: " + e.message;
        }}
    }}, 100);
</script>
</body>
</html>
"""
    with open(os.path.join(OUT_DIR, "graph.html"), "w") as f:
        f.write(html)
        
    with open(CACHE_FILE, "w") as f:
        json.dump({"db_hash": current_hash}, f)
        
    print("Graphify completed successfully! Check the graphify-out/ folder.")

if __name__ == "__main__":
    main()
