"""
Database Schema:
    Users
        - id: int
        - username: str
        - repos: List[int]
        - merged_trees: List[int]
    Repos:
        - id: int
        - name: str
        - files: List[int]
    Files:
        - id: int
        - url: str
        - trees: List[int]
    Nodes:
        - id: int
        - name: str
        - is_callable: bool
        - order: int
        - frequency: int
        - children: List[int]
        - is_merged: bool
    Modules:
        - id: int
        - name: str
        - merged_tree: int
        - top_users: List[int]
"""

import os
import requests
import urllib.request

from pymongo import MongoClient
from saplings import Saplings

GH_API_BASE = "https://api.github.com"
GH_RAW_BASE = "https://raw.githubusercontent.com"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CLIENT = MongoClient("mongodb://localhost:27017/saplings")


def _download_file(file_url):
    unique_filename = str(abs(hash(file_url)))
    local_filepath = os.path.join(CACHE_DIR, f"{unique_filename}.py")
    urllib.request.urlretrieve(file_url, local_file_path)

    return local_file_path


def _add_tree_to_db(node):
    record = {
        "name": node.name,
        "is_callable": node.is_callable,
        "order": node.order,
        "frequency": node.frequency,
        "children": []
    }
    for child in node.children:
        record["children"].append(_add_tree_to_db(child))

    # TODO: Insert record into DB and return ID
    return


def _extract_module_trees(filepath):
    try:
        ast_root = ast.parse(open(filepath, "r").read())
    except:
        pass # TODO: Error-handling (what should you write to DB?)
    finally:
        os.remove(filepath)

    trees = Saplings(ast_root).get_trees()
    tree_ids = [_add_tree_to_db(tree) for tree in trees]

    return tree_ids


def extract_module_trees_from_repo(username, repo):
    # NOTE: We ignore forked repos
    repo_url = f"{GH_API_BASE}/repos/{username}/{repo}"
    response = requests.get(repo_url).json()

    master = response["default_branch"]
    response = requests.get(f"{repo_url}/git/trees/{master}?recursive=1").json()

    # TODO: Parallelize (threads since I/O bound)
    for file in response["tree"]:
        filepath = file["path"]
        file_url = f"{GH_RAW_BASE}/{username}/{repo}/{master}/{filepath}"

        local_filepath = _download_file(file_url)
        tree_ids = _extract_module_trees(local_filepath)


def _unroll_tree_from_root(root_id):
    root = DB.nodes.find_one({"id": root_id})
    children = []
    for child_id in root["children"]:
        child_tree = _unroll_tree_from_root(child_id)
        children.append(child_tree)

    tree = {
        root["name"]: {
            "is_callable": root["is_callable"],
            "order": root["order"],
            "frequency": root["frequency"],
            "children": children
        }
    }
    return tree


def _get_trees_by_user(username):
    user = DB.users.find_one({"username": username})
    for repo_id in user["repos"]:
        repo = DB.repos.find_one({"id": repo_id})
        for file_id in repo["files"]:
            file = DB.files.find_one({"id": file_id})
            for node_id in file["trees"]:
                yield DB.nodes.find_one({"id": node_id})


def _merge_nodes(merged_nodes, nodes):
    for node_id in nodes:
        node = DB.nodes.find_one({"id": node_id})
        if node["name"] in merged_nodes: # TODO: Check by name and order
            merged_nodes[node["name"]]["frequency"] += 1
        else:
            merged_nodes[node["name"]] = {
                "is_callable": node["is_callable"],
                "order": node["order"],
                "frequency": node["frequency"],
                "children": {}
            }

        for child_id in node["children"]:
            _merge_nodes(merged_nodes[node["name"]]["children"], child_id)



def merge_trees_by_user(username):
    nodes = {}
    for node in _get_trees_by_user(username):






def merge_trees_by_module():
    pass
