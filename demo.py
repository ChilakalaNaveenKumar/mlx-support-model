# universal_ast_generator.py with CLI + PNG Graph Output
import os
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from tree_sitter_languages import get_parser

EXT_TO_LANG = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.tsx': 'typescript', '.jsx': 'javascript',
    '.java': 'java', '.c': 'c', '.cpp': 'cpp', '.cc': 'cpp', '.go': 'go', '.php': 'php',
    '.rb': 'ruby', '.json': 'json', '.html': 'html', '.css': 'css'
}

PROJECT_SIGNATURES = {
    'django': lambda files: any('manage.py' in f or 'settings.py' in f for f in files),
    'flask': lambda files: any('app.py' in f or 'run.py' in f for f in files),
    'fastapi': lambda files: any(f.endswith('.py') and 'FastAPI' in open(f).read() for f in files if os.path.exists(f)),
    'react': lambda files: 'package.json' in files and 'react' in open('package.json').read(),
    'nextjs': lambda files: 'next.config.js' in files or 'next.config.mjs' in files,
    'vue': lambda files: any(f.endswith('.vue') for f in files),
    'nuxt': lambda files: 'nuxt.config.js' in files,
    'angular': lambda files: 'angular.json' in files,
    'svelte': lambda files: any(f.endswith('.svelte') for f in files),
    'spring': lambda files: 'pom.xml' in files or 'build.gradle' in files,
    'express': lambda files: any(f in files for f in ['app.js', 'server.js', 'index.js']),
    'node': lambda files: 'package.json' in files and 'node_modules' in os.listdir(),
    'static-site': lambda files: any(f.endswith(('.html', '.css', '.js')) for f in files),
    'json-api': lambda files: any(f.endswith('.json') for f in files),
}

def detect_project_type(files):
    for proj, check in PROJECT_SIGNATURES.items():
        try:
            if check(files):
                return proj
        except:
            continue
    return 'generic'

def generate_ast(filepath, parser):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    tree = parser.parse(code.encode("utf8"))
    def node_to_dict(node):
        return {
            "type": node.type,
            "start": node.start_point,
            "end": node.end_point,
            "children": [node_to_dict(child) for child in node.children]
        }
    return node_to_dict(tree.root_node)

def build_graph(ast_dict, graph=None, parent=None):
    if graph is None:
        graph = nx.DiGraph()
    node_id = f"{ast_dict['type']}_{id(ast_dict)}"
    graph.add_node(node_id, type=ast_dict['type'])
    if parent:
        graph.add_edge(parent, node_id)
    for child in ast_dict.get('children', []):
        build_graph(child, graph, node_id)
    return graph

def visualize_graph(gml_path, png_path):
    G = nx.read_gml(gml_path)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=30, font_size=6)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['type'] for n in G.nodes}, font_size=6)
    plt.title("AST Knowledge Graph")
    plt.savefig(png_path)
    plt.close()
    print(f"[Graph] Saved visualization to {png_path}")

def main(directory, visualize=False, specific_lang=None):
    available_parsers = {}
    for lang in set(EXT_TO_LANG.values()):
        try:
            parser = get_parser(lang)
            if parser:
                available_parsers[lang] = parser
        except:
            continue

    for root, _, files in os.walk(directory):
        full_paths = [os.path.join(root, f) for f in files]
        detected = detect_project_type(full_paths)
        print(f"[Project Type] {root} → {detected}")

        for file in files:
            ext = os.path.splitext(file)[1]
            lang = EXT_TO_LANG.get(ext)
            if not lang or lang not in available_parsers:
                continue
            if specific_lang and lang != specific_lang:
                continue

            full_path = os.path.join(root, file)
            try:
                parser = available_parsers[lang]
                ast_tree = generate_ast(full_path, parser)

                json_path = f"{full_path}.ast.json"
                with open(json_path, "w") as out:
                    json.dump(ast_tree, out, indent=2)

                graph = build_graph(ast_tree)
                gml_path = f"{full_path}.gml"
                nx.write_gml(graph, gml_path)
                print(f"Parsed {file} ({lang}) → AST + Graph saved")

                if visualize:
                    png_path = f"{full_path}.png"
                    visualize_graph(gml_path, png_path)

            except Exception as e:
                print(f"[ERROR] Failed to parse {file}: {e}")

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Universal AST Generator with Knowledge Graphs")
    parser.add_argument("--dir", default="sample_project", help="Path to codebase directory")
    parser.add_argument("--lang", default=None, help="Restrict to a specific language (e.g. 'python')")
    parser.add_argument("--visualize", action="store_true", help="Generate PNG visualization from AST graph")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"[ERROR] Provided directory '{args.dir}' does not exist.")
        sys.exit(1)

    demo_path = os.path.join(args.dir, "demo.py")
    if not os.path.exists(demo_path):
        print(f"[ERROR] File '{demo_path}' not found. Please create it or specify a different file.")
        sys.exit(1)

    print(f"Running on: {args.dir}")
    main(args.dir, visualize=args.visualize, specific_lang=args.lang)

    print(f"Running on: {args.dir}")
    main(args.dir, visualize=args.visualize, specific_lang=args.lang)
