#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A comprehensive Python application demonstrating various programming concepts,
data structures, algorithms, and practical implementations.

This application includes:
1. Data structures implementations
2. Various algorithms and their implementations
3. Design patterns
4. Web application components
5. Machine learning examples
6. Data processing utilities
7. Testing frameworks
8. Network programming examples
9. File system operations
10. Database operations
"""

import abc
import argparse
import asyncio
import base64
import calendar
import collections
import concurrent.futures
import contextlib
import copy
import csv
import dataclasses
import datetime
import decimal
import enum
import functools
import glob
import gzip
import hashlib
import heapq
import hmac
import html
import http.client
import http.cookiejar
import http.server
import importlib
import inspect
import io
import itertools
import json
import logging
import math
import mimetypes
import multiprocessing
import operator
import os
import pathlib
import pickle
import platform
import pprint
import queue
import random
import re
import secrets
import select
import shelve
import shutil
import signal
import socket
import socketserver
import sqlite3
import statistics
import string
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import timeit
import types
import typing
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
import weakref
import xml.etree.ElementTree as ET
import zipfile
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, Iterator, Generator
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


#############################################################################
# SECTION 1: DATA STRUCTURES
#############################################################################

class Node:
    """A basic node for linked data structures."""
    
    def __init__(self, data=None):
        self.data = data
        self.next = None
        
    def __str__(self):
        return f"Node({self.data})"


class LinkedList:
    """An implementation of a singly linked list."""
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
        
    def append(self, data):
        """Add a new node with given data to the end of the list."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
        
    def prepend(self, data):
        """Add a new node with given data to the beginning of the list."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self.size += 1
        
    def delete(self, data):
        """Remove the first occurrence of data in the list."""
        if self.head is None:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            self.size -= 1
            if self.head is None:
                self.tail = None
            return
        
        current = self.head
        while current.next and current.next.data != data:
            current = current.next
            
        if current.next:
            if current.next == self.tail:
                self.tail = current
            current.next = current.next.next
            self.size -= 1
            
    def find(self, data):
        """Find if data exists in the list."""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False
    
    def get_at_index(self, index):
        """Get the node at the given index."""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        
        current = self.head
        for _ in range(index):
            current = current.next
        return current.data
    
    def insert_at_index(self, index, data):
        """Insert a new node with data at the given index."""
        if index < 0 or index > self.size:
            raise IndexError("Index out of range")
        
        if index == 0:
            self.prepend(data)
            return
        
        if index == self.size:
            self.append(data)
            return
        
        new_node = Node(data)
        current = self.head
        for _ in range(index - 1):
            current = current.next
            
        new_node.next = current.next
        current.next = new_node
        self.size += 1
        
    def __len__(self):
        return self.size
    
    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next
            
    def __str__(self):
        values = [str(x) for x in self]
        return ' -> '.join(values)


class DoublyLinkedNode:
    """A node for a doubly linked list."""
    
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None
        
    def __str__(self):
        return f"DoublyLinkedNode({self.data})"


class DoublyLinkedList:
    """An implementation of a doubly linked list."""
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
        
    def append(self, data):
        """Add a new node with given data to the end of the list."""
        new_node = DoublyLinkedNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
        
    def prepend(self, data):
        """Add a new node with given data to the beginning of the list."""
        new_node = DoublyLinkedNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1
        
    def delete(self, data):
        """Remove the first occurrence of data in the list."""
        if self.head is None:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            else:
                self.tail = None
            self.size -= 1
            return
        
        if self.tail.data == data:
            self.tail = self.tail.prev
            self.tail.next = None
            self.size -= 1
            return
        
        current = self.head
        while current and current.data != data:
            current = current.next
            
        if current:
            current.prev.next = current.next
            if current.next:
                current.next.prev = current.prev
            self.size -= 1
            
    def find(self, data):
        """Find if data exists in the list."""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False
    
    def get_at_index(self, index):
        """Get the node at the given index."""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        
        if index < self.size // 2:
            current = self.head
            for _ in range(index):
                current = current.next
        else:
            current = self.tail
            for _ in range(self.size - 1, index, -1):
                current = current.prev
        
        return current.data
    
    def insert_at_index(self, index, data):
        """Insert a new node with data at the given index."""
        if index < 0 or index > self.size:
            raise IndexError("Index out of range")
        
        if index == 0:
            self.prepend(data)
            return
        
        if index == self.size:
            self.append(data)
            return
        
        new_node = DoublyLinkedNode(data)
        
        if index < self.size // 2:
            current = self.head
            for _ in range(index - 1):
                current = current.next
            
            new_node.next = current.next
            new_node.prev = current
            current.next.prev = new_node
            current.next = new_node
        else:
            current = self.tail
            for _ in range(self.size - 1, index, -1):
                current = current.prev
                
            new_node.prev = current.prev
            new_node.next = current
            current.prev.next = new_node
            current.prev = new_node
            
        self.size += 1
        
    def __len__(self):
        return self.size
    
    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next
            
    def __str__(self):
        values = [str(x) for x in self]
        return ' <-> '.join(values)


class Stack:
    """An implementation of a stack (LIFO) data structure."""
    
    def __init__(self):
        self.items = []
        
    def push(self, item):
        """Add an item to the top of the stack."""
        self.items.append(item)
        
    def pop(self):
        """Remove and return the top item from the stack."""
        if self.is_empty():
            raise IndexError("Pop from an empty stack")
        return self.items.pop()
    
    def peek(self):
        """Return the top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek at an empty stack")
        return self.items[-1]
    
    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the stack."""
        return len(self.items)
    
    def __len__(self):
        return self.size()
    
    def __str__(self):
        return str(self.items)


class Queue:
    """An implementation of a queue (FIFO) data structure."""
    
    def __init__(self):
        self.items = collections.deque()
        
    def enqueue(self, item):
        """Add an item to the end of the queue."""
        self.items.append(item)
        
    def dequeue(self):
        """Remove and return the front item from the queue."""
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        return self.items.popleft()
    
    def peek(self):
        """Return the front item without removing it."""
        if self.is_empty():
            raise IndexError("Peek at an empty queue")
        return self.items[0]
    
    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        return len(self.items)
    
    def __len__(self):
        return self.size()
    
    def __str__(self):
        return str(list(self.items))


class Deque:
    """An implementation of a double-ended queue."""
    
    def __init__(self):
        self.items = collections.deque()
        
    def add_front(self, item):
        """Add an item to the front of the deque."""
        self.items.appendleft(item)
        
    def add_rear(self, item):
        """Add an item to the rear of the deque."""
        self.items.append(item)
        
    def remove_front(self):
        """Remove and return the front item from the deque."""
        if self.is_empty():
            raise IndexError("Remove from an empty deque")
        return self.items.popleft()
    
    def remove_rear(self):
        """Remove and return the rear item from the deque."""
        if self.is_empty():
            raise IndexError("Remove from an empty deque")
        return self.items.pop()
    
    def peek_front(self):
        """Return the front item without removing it."""
        if self.is_empty():
            raise IndexError("Peek at an empty deque")
        return self.items[0]
    
    def peek_rear(self):
        """Return the rear item without removing it."""
        if self.is_empty():
            raise IndexError("Peek at an empty deque")
        return self.items[-1]
    
    def is_empty(self):
        """Check if the deque is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the deque."""
        return len(self.items)
    
    def __len__(self):
        return self.size()
    
    def __str__(self):
        return str(list(self.items))


class BinaryTreeNode:
    """A node for a binary tree."""
    
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None
        
    def __str__(self):
        return f"BinaryTreeNode({self.data})"


class BinarySearchTree:
    """An implementation of a binary search tree."""
    
    def __init__(self):
        self.root = None
        self.size = 0
        
    def insert(self, data):
        """Insert a new node with data into the BST."""
        if self.root is None:
            self.root = BinaryTreeNode(data)
            self.size += 1
            return
        
        self._insert_recursive(self.root, data)
        
    def _insert_recursive(self, node, data):
        """Recursively insert data into the BST."""
        if data < node.data:
            if node.left is None:
                node.left = BinaryTreeNode(data)
                self.size += 1
            else:
                self._insert_recursive(node.left, data)
        elif data > node.data:
            if node.right is None:
                node.right = BinaryTreeNode(data)
                self.size += 1
            else:
                self._insert_recursive(node.right, data)
        # If data is equal to node.data, we do nothing (no duplicates)
    
    def search(self, data):
        """Search for data in the BST."""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node, data):
        """Recursively search for data in the BST."""
        if node is None:
            return False
        
        if data == node.data:
            return True
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)
            
    def delete(self, data):
        """Delete a node with data from the BST."""
        self.root, deleted = self._delete_recursive(self.root, data)
        if deleted:
            self.size -= 1
            
    def _delete_recursive(self, node, data):
        """Recursively delete a node with data from the BST."""
        if node is None:
            return None, False
        
        if data < node.data:
            node.left, deleted = self._delete_recursive(node.left, data)
            return node, deleted
        elif data > node.data:
            node.right, deleted = self._delete_recursive(node.right, data)
            return node, deleted
        else:
            # Case 1: Node has no children
            if node.left is None and node.right is None:
                return None, True
            
            # Case 2: Node has only one child
            if node.left is None:
                return node.right, True
            if node.right is None:
                return node.left, True
            
            # Case 3: Node has two children
            # Find the inorder successor (smallest node in right subtree)
            successor = self._find_min(node.right)
            node.data = successor.data
            node.right, _ = self._delete_recursive(node.right, successor.data)
            return node, True
            
    def _find_min(self, node):
        """Find the node with the minimum value in the subtree."""
        current = node
        while current.left:
            current = current.left
        return current
    
    def inorder_traversal(self):
        """Return a list of all nodes in inorder traversal."""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        """Helper method for inorder traversal."""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)
            
    def preorder_traversal(self):
        """Return a list of all nodes in preorder traversal."""
        result = []
        self._preorder_recursive(self.root, result)
        return result
    
    def _preorder_recursive(self, node, result):
        """Helper method for preorder traversal."""
        if node:
            result.append(node.data)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)
            
    def postorder_traversal(self):
        """Return a list of all nodes in postorder traversal."""
        result = []
        self._postorder_recursive(self.root, result)
        return result
    
    def _postorder_recursive(self, node, result):
        """Helper method for postorder traversal."""
        if node:
            self._postorder_recursive(node.left, result)
            self._postorder_recursive(node.right, result)
            result.append(node.data)
            
    def __len__(self):
        return self.size
    
    def __str__(self):
        return str(self.inorder_traversal())


class AVLNode(BinaryTreeNode):
    """A node for an AVL tree that keeps track of its height."""
    
    def __init__(self, data=None):
        super().__init__(data)
        self.height = 1
        
    def __str__(self):
        return f"AVLNode({self.data}, h={self.height})"


class AVLTree:
    """An implementation of a self-balancing AVL tree."""
    
    def __init__(self):
        self.root = None
        self.size = 0
        
    def height(self, node):
        """Get the height of a node."""
        if node is None:
            return 0
        return node.height
    
    def balance_factor(self, node):
        """Calculate the balance factor of a node."""
        if node is None:
            return 0
        return self.height(node.left) - self.height(node.right)
    
    def update_height(self, node):
        """Update the height of a node based on its children."""
        if node is None:
            return
        node.height = 1 + max(self.height(node.left), self.height(node.right))
        
    def right_rotate(self, y):
        """Perform a right rotation on node y."""
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        self.update_height(y)
        self.update_height(x)
        
        return x
    
    def left_rotate(self, x):
        """Perform a left rotation on node x."""
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        self.update_height(x)
        self.update_height(y)
        
        return y
        
    def insert(self, data):
        """Insert a new node with data into the AVL tree."""
        self.root = self._insert_recursive(self.root, data)
        
    def _insert_recursive(self, node, data):
        """Recursively insert data into the AVL tree."""
        # Perform standard BST insert
        if node is None:
            self.size += 1
            return AVLNode(data)
        
        if data < node.data:
            node.left = self._insert_recursive(node.left, data)
        elif data > node.data:
            node.right = self._insert_recursive(node.right, data)
        else:
            # Duplicate data not allowed
            return node
        
        # Update height of this ancestor node
        self.update_height(node)
        
        # Get the balance factor to check if this node became unbalanced
        balance = self.balance_factor(node)
        
        # If the node becomes unbalanced, there are four cases
        
        # Left Left Case
        if balance > 1 and data < node.left.data:
            return self.right_rotate(node)
        
        # Right Right Case
        if balance < -1 and data > node.right.data:
            return self.left_rotate(node)
        
        # Left Right Case
        if balance > 1 and data > node.left.data:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        
        # Right Left Case
        if balance < -1 and data < node.right.data:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)
        
        # Return the unchanged node
        return node
    
    def delete(self, data):
        """Delete a node with data from the AVL tree."""
        if self.root is None:
            return
        
        self.root = self._delete_recursive(self.root, data)
        
    def _delete_recursive(self, node, data):
        """Recursively delete a node with data from the AVL tree."""
        # Perform standard BST delete
        if node is None:
            return None
        
        if data < node.data:
            node.left = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right = self._delete_recursive(node.right, data)
        else:
            # Node with only one child or no children
            if node.left is None:
                temp = node.right
                node = None
                self.size -= 1
                return temp
            elif node.right is None:
                temp = node.left
                node = None
                self.size -= 1
                return temp
            
            # Node with two children
            temp = self._get_min_value_node(node.right)
            node.data = temp.data
            node.right = self._delete_recursive(node.right, temp.data)
        
        # If the tree has only one node
        if node is None:
            return None
        
        # Update height of the current node
        self.update_height(node)
        
        # Get the balance factor to check if this node became unbalanced
        balance = self.balance_factor(node)
        
        # If the node becomes unbalanced, there are four cases
        
        # Left Left Case
        if balance > 1 and self.balance_factor(node.left) >= 0:
            return self.right_rotate(node)
        
        # Left Right Case
        if balance > 1 and self.balance_factor(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        
        # Right Right Case
        if balance < -1 and self.balance_factor(node.right) <= 0:
            return self.left_rotate(node)
        
        # Right Left Case
        if balance < -1 and self.balance_factor(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)
        
        return node
    
    def _get_min_value_node(self, node):
        """Find the node with the minimum value in the subtree."""
        current = node
        while current.left is not None:
            current = current.left
        return current
    
    def search(self, data):
        """Search for data in the AVL tree."""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node, data):
        """Recursively search for data in the AVL tree."""
        if node is None:
            return False
        
        if data == node.data:
            return True
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)
            
    def inorder_traversal(self):
        """Return a list of all nodes in inorder traversal."""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        """Helper method for inorder traversal."""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)
            
    def __len__(self):
        return self.size
    
    def __str__(self):
        return str(self.inorder_traversal())


class MinHeap:
    """An implementation of a min heap (priority queue)."""
    
    def __init__(self):
        self.heap = []
        
    def parent(self, i):
        """Return the index of the parent of node at index i."""
        return (i - 1) // 2
    
    def left_child(self, i):
        """Return the index of the left child of node at index i."""
        return 2 * i + 1
    
    def right_child(self, i):
        """Return the index of the right child of node at index i."""
        return 2 * i + 2
    
    def get_min(self):
        """Return the minimum element (root) without removing it."""
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def extract_min(self):
        """Remove and return the minimum element (root) from the heap."""
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        
        min_item = self.heap[0]
        self.heap[0] = self.heap[-1]
        del self.heap[-1]
        self._heapify_down(0)
        
        return min_item
    
    def insert(self, item):
        """Insert a new item into the heap."""
        self.heap.append(item)
        self._heapify_up(len(self.heap) - 1)
        
    def _heapify_up(self, i):
        """Restore heap property by moving element up."""
        parent_idx = self.parent(i)
        
        if i > 0 and self.heap[i] < self.heap[parent_idx]:
            self.heap[i], self.heap[parent_idx] = self.heap[parent_idx], self.heap[i]
            self._heapify_up(parent_idx)
            
    def _heapify_down(self, i):
        """Restore heap property by moving element down."""
        smallest = i
        left = self.left_child(i)
        right = self.right_child(i)
        n = len(self.heap)
        
        if left < n and self.heap[left] < self.heap[smallest]:
            smallest = left
            
        if right < n and self.heap[right] < self.heap[smallest]:
            smallest = right
            
        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self._heapify_down(smallest)
    
    def build_heap(self, arr):
        """Build a heap from an array."""
        self.heap = arr.copy()
        n = len(self.heap)
        
        # Start from the last non-leaf node and heapify down
        for i in range(n // 2 - 1, -1, -1):
            self._heapify_down(i)
            
    def is_empty(self):
        """Check if the heap is empty."""
        return len(self.heap) == 0
    
    def size(self):
        """Return the number of items in the heap."""
        return len(self.heap)
    
    def __len__(self):
        return self.size()
    
    def __str__(self):
        return str(self.heap)


class MaxHeap(MinHeap):
    """An implementation of a max heap."""
    
    def _heapify_up(self, i):
        """Restore heap property by moving element up."""
        parent_idx = self.parent(i)
        
        if i > 0 and self.heap[i] > self.heap[parent_idx]:
            self.heap[i], self.heap[parent_idx] = self.heap[parent_idx], self.heap[i]
            self._heapify_up(parent_idx)
            
    def _heapify_down(self, i):
        """Restore heap property by moving element down."""
        largest = i
        left = self.left_child(i)
        right = self.right_child(i)
        n = len(self.heap)
        
        if left < n and self.heap[left] > self.heap[largest]:
            largest = left
            
        if right < n and self.heap[right] > self.heap[largest]:
            largest = right
            
        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self._heapify_down(largest)
            
    def get_max(self):
        """Return the maximum element (root) without removing it."""
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def extract_max(self):
        """Remove and return the maximum element (root) from the heap."""
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        
        max_item = self.heap[0]
        self.heap[0] = self.heap[-1]
        del self.heap[-1]
        self._heapify_down(0)
        
        return max_item


class Graph:
    """An implementation of an undirected graph using an adjacency list."""
    
    def __init__(self):
        self.adjacency_list = {}
        
    def add_vertex(self, vertex):
        """Add a vertex to the graph."""
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
            
    def add_edge(self, vertex1, vertex2, weight=1):
        """Add an edge between two vertices."""
        if vertex1 not in self.adjacency_list:
            self.add_vertex(vertex1)
        if vertex2 not in self.adjacency_list:
            self.add_vertex(vertex2)
            
        # Add edge in both directions for undirected graph
        self.adjacency_list[vertex1].append((vertex2, weight))
        self.adjacency_list[vertex2].append((vertex1, weight))
        
    def remove_edge(self, vertex1, vertex2):
        """Remove the edge between two vertices."""
        if vertex1 in self.adjacency_list and vertex2 in self.adjacency_list:
            self.adjacency_list[vertex1] = [edge for edge in self.adjacency_list[vertex1] if edge[0] != vertex2]
            self.adjacency_list[vertex2] = [edge for edge in self.adjacency_list[vertex2] if edge[0] != vertex1]
            
    def remove_vertex(self, vertex):
        """Remove a vertex and all its edges from the graph."""
        if vertex in self.adjacency_list:
            # Remove all edges associated with this vertex
            for other_vertex in self.adjacency_list:
                self.adjacency_list[other_vertex] = [edge for edge in self.adjacency_list[other_vertex] if edge[0] != vertex]
            
            # Remove the vertex itself
            del self.adjacency_list[vertex]
            
    def get_vertices(self):
        """Return all vertices in the graph."""
        return list(self.adjacency_list.keys())
    
    def get_edges(self):
        """Return all edges in the graph as (vertex1, vertex2, weight) tuples."""
        edges = []
        for vertex in self.adjacency_list:
            for neighbor, weight in self.adjacency_list[vertex]:
                if (neighbor, vertex, weight) not in edges:  # Avoid duplicates due to undirected graph
                    edges.append((vertex, neighbor, weight))
        return edges
    
    def get_neighbors(self, vertex):
        """Return all neighbors of a vertex."""
        if vertex in self.adjacency_list:
            return [edge[0] for edge in self.adjacency_list[vertex]]
        return []
    
    def breadth_first_search(self, start_vertex):
        """Perform a breadth-first search starting from start_vertex."""
        if start_vertex not in self.adjacency_list:
            return []
            
        result = []
        visited = set()
        queue = collections.deque([start_vertex])
        visited.add(start_vertex)
        
        while queue:
            current_vertex = queue.popleft()
            result.append(current_vertex)
            
            for neighbor, _ in self.adjacency_list[current_vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return result
    
    def depth_first_search(self, start_vertex):
        """Perform a depth-first search starting from start_vertex."""
        if start_vertex not in self.adjacency_list:
            return []
            
        result = []
        visited = set()
        
        def dfs_helper(vertex):
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor, _ in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    dfs_helper(neighbor)
                    
        dfs_helper(start_vertex)
        return result
    
    def dijkstra(self, start_vertex):
        """Run Dijkstra's algorithm to find the shortest path from start_vertex to all other vertices."""
        if start_vertex not in self.adjacency_list:
            return {}
            
        distances = {vertex: float('infinity') for vertex in self.adjacency_list}
        distances[start_vertex] = 0
        priority_queue = [(0, start_vertex)]
        
        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)
            
            # Skip if we've found a better path already
            if current_distance > distances[current_vertex]:
                continue
                
            for neighbor, weight in self.adjacency_list[current_vertex]:
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
                    
        return distances
    
    def __str__(self):
        result = ""
        for vertex in self.adjacency_list:
            result += f"{vertex} -> {self.adjacency_list[vertex]}\n"
        return result


class DisjointSet:
    """An implementation of a disjoint-set data structure (Union-Find)."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
        
    def make_set(self, item):
        """Create a new set with a single element."""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            
    def find(self, item):
        """Find the representative (root) of a set containing item."""
        if item not in self.parent:
            return None
            
        if self.parent[item] != item:
            # Path compression: update parent to be the root
            self.parent[item] = self.find(self.parent[item])
            
        return self.parent[item]
        
    def union(self, item1, item2):
        """Merge the sets containing item1 and item2."""
        root1 = self.find(item1)
        root2 = self.find(item2)
        
        if root1 is None or root2 is None:
            return
            
        if root1 == root2:
            return
            
        # Union by rank: attach smaller rank tree under root of higher rank tree
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1
            
    def is_same_set(self, item1, item2):
        """Check if item1 and item2 are in the same set."""
        return self.find(item1) == self.find(item2)
    
    def get_sets(self):
        """Return all sets as a dictionary {representative: set_elements}."""
        sets = {}
        for item in self.parent:
            rep = self.find(item)
            if rep not in sets:
                sets[rep] = []
            sets[rep].append(item)
        return sets
    
    def __str__(self):
        sets = self.get_sets()
        result = ""
        for rep, items in sets.items():
            result += f"{rep}: {items}\n"
        return result


class Trie:
    """An implementation of a trie (prefix tree)."""
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
            
    def __init__(self):
        self.root = self.TrieNode()
        
    def insert(self, word):
        """Insert a word into the trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        
    def search(self, word):
        """Search for a complete word in the trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        """Check if there is any word in the trie that starts with the given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def get_words_with_prefix(self, prefix):
        """Get all words in the trie that start with the given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
            
        words = []
        self._collect_words(node, prefix, words)
        return words
    
    def _collect_words(self, node, prefix, words):
        """Helper method to collect words with a common prefix."""
        if node.is_end_of_word:
            words.append(prefix)
            
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, words)
            
    def delete(self, word):
        """Delete a word from the trie."""
        self._delete_recursive(self.root, word, 0)
        
    def _delete_recursive(self, node, word, index):
        """Helper method to recursively delete a word."""
        # Base case: reached the end of the word
        if index == len(word):
            if node.is_end_of_word:
                node.is_end_of_word = False
            return len(node.children) == 0
            
        char = word[index]
        if char not in node.children:
            return False
            
        should_delete_node = self._delete_recursive(node.children[char], word, index + 1)
        
        if should_delete_node:
            del node.children[char]
            return len(node.children) == 0 and not node.is_end_of_word
            
        return False


#############################################################################
# SECTION 2: ALGORITHMS
#############################################################################

class SortingAlgorithms:
    """A collection of sorting algorithms."""
    
    @staticmethod
    def bubble_sort(arr):
        """Implementation of bubble sort algorithm."""
        n = len(arr)
        for i in range(n):
            # Flag to optimize if array is already sorted
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            # If no swapping occurred in this pass, array is sorted
            if not swapped:
                break
        return arr
    
    @staticmethod
    def selection_sort(arr):
        """Implementation of selection sort algorithm."""
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr
    
    @staticmethod
    def insertion_sort(arr):
        """Implementation of insertion sort algorithm."""
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    @staticmethod
    def merge_sort(arr):
        """Implementation of merge sort algorithm."""
        if len(arr) <= 1:
            return arr
            
        mid = len(arr) // 2
        left = SortingAlgorithms.merge_sort(arr[:mid])
        right = SortingAlgorithms.merge_sort(arr[mid:])
        
        return SortingAlgorithms._merge(left, right)
    
    @staticmethod
    def _merge(left, right):
        """Helper method to merge two sorted arrays."""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
                
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    @staticmethod
    def quick_sort(arr):
        """Implementation of quick sort algorithm."""
        SortingAlgorithms._quick_sort_helper(arr, 0, len(arr) - 1)
        return arr
    
    @staticmethod
    def _quick_sort_helper(arr, low, high):
        """Helper method for quick sort."""
        if low < high:
            pivot_idx = SortingAlgorithms._partition(arr, low, high)
            SortingAlgorithms._quick_sort_helper(arr, low, pivot_idx - 1)
            SortingAlgorithms._quick_sort_helper(arr, pivot_idx + 1, high)
    
    @staticmethod
    def _partition(arr, low, high):
        """Helper method to partition the array for quick sort."""
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    @staticmethod
    def heap_sort(arr):
        """Implementation of heap sort algorithm."""
        n = len(arr)
        
        # Build a max heap
        for i in range(n // 2 - 1, -1, -1):
            SortingAlgorithms._heapify(arr, n, i)
            
        # Extract elements one by one
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            SortingAlgorithms._heapify(arr, i, 0)
            
        return arr
    
    @staticmethod
    def _heapify(arr, n, i):
        """Helper method to heapify a subtree rooted at index i."""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] > arr[largest]:
            largest = left
            
        if right < n and arr[right] > arr[largest]:
            largest = right
            
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            SortingAlgorithms._heapify(arr, n, largest)
            
    @staticmethod
    def counting_sort(arr, max_val=None):
        """Implementation of counting sort algorithm."""
        if not arr:
            return arr
            
        if max_val is None:
            max_val = max(arr)
            
        count = [0] * (max_val + 1)
        
        # Count occurrences
        for num in arr:
            count[num] += 1
            
        # Reconstruct the sorted array
        sorted_arr = []
        for i in range(len(count)):
            sorted_arr.extend([i] * count[i])
            
        return sorted_arr
    
    @staticmethod
    def radix_sort(arr):
        """Implementation of radix sort algorithm for non-negative integers."""
        if not arr:
            return arr
            
        # Find the maximum number to know the number of digits
        max_val = max(arr)
        exp = 1
        
        while max_val // exp > 0:
            SortingAlgorithms._counting_sort_by_digit(arr, exp)
            exp *= 10
            
        return arr
    
    @staticmethod
    def _counting_sort_by_digit(arr, exp):
        """Helper method for radix sort to sort by a specific digit."""
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        
        # Count occurrences of each digit
        for i in range(n):
            index = arr[i] // exp % 10
            count[index] += 1
            
        # Change count[i] so that it contains actual position of this digit in output
        for i in range(1, 10):
            count[i] += count[i - 1]
            
        # Build the output array
        for i in range(n - 1, -1, -1):
            index = arr[i] // exp % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            
        # Copy the output array to arr
        for i in range(n):
            arr[i] = output[i]


class SearchAlgorithms:
    """A collection of search algorithms."""
    
    @staticmethod
    def linear_search(arr, target):
        """Implementation of linear search algorithm."""
        for i, item in enumerate(arr):
            if item == target:
                return i
        return -1
    
    @staticmethod
    def binary_search(arr, target):
        """Implementation of binary search algorithm (assumes sorted array)."""
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
                
        return -1
    
    @staticmethod
    def binary_search_recursive(arr, target):
        """Recursive implementation of binary search algorithm."""
        return SearchAlgorithms._binary_search_helper(arr, target, 0, len(arr) - 1)
    
    @staticmethod
    def _binary_search_helper(arr, target, left, right):
        """Helper method for recursive binary search."""
        if left > right:
            return -1
            
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return SearchAlgorithms._binary_search_helper(arr, target, mid + 1, right)
        else:
            return SearchAlgorithms._binary_search_helper(arr, target, left, mid - 1)
    
    @staticmethod
    def jump_search(arr, target):
        """Implementation of jump search algorithm (assumes sorted array)."""
        n = len(arr)
        step = int(math.sqrt(n))
        
        # Finding the block where the target may be present
        prev = 0
        while prev < n and arr[min(step, n) - 1] < target:
            prev = step
            step += int(math.sqrt(n))
            if prev >= n:
                return -1
                
        # Linear search in the identified block
        while prev < min(step, n):
            if arr[prev] == target:
                return prev
            prev += 1
            
        return -1
    
    @staticmethod
    def interpolation_search(arr, target):
        """Implementation of interpolation search algorithm (assumes sorted array)."""
        left, right = 0, len(arr) - 1
        
        while left <= right and arr[left] <= target <= arr[right]:
            if left == right:
                if arr[left] == target:
                    return left
                return -1
                
            # Formula for interpolation point
            pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])
            
            if arr[pos] == target:
                return pos
            elif arr[pos] < target:
                left = pos + 1
            else:
                right = pos - 1
                
        return -1
    
    @staticmethod
    def exponential_search(arr, target):
        """Implementation of exponential search algorithm (assumes sorted array)."""
        n = len(arr)
        
        if n == 0:
            return -1
            
        if arr[0] == target:
            return 0
            
        # Find range for binary search
        i = 1
        while i < n and arr[i] <= target:
            i *= 2
            
        # Call binary search for the found range
        return SearchAlgorithms._binary_search_helper(arr, target, i // 2, min(i, n - 1))


class StringAlgorithms:
    """A collection of string processing algorithms."""
    
    @staticmethod
    def naive_string_matching(text, pattern):
        """Implementation of naive string matching algorithm."""
        n, m = len(text), len(pattern)
        results = []
        
        for i in range(n - m + 1):
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                results.append(i)
                
        return results
    
    @staticmethod
    def kmp_string_matching(text, pattern):
        """Implementation of Knuth-Morris-Pratt (KMP) string matching algorithm."""
        if not pattern:
            return [i for i in range(len(text) + 1)]
            
        # Compute the failure function
        failure = [0] * len(pattern)
        j = 0
        
        for i in range(1, len(pattern)):
            while j > 0 and pattern[j] != pattern[i]:
                j = failure[j - 1]
            if pattern[j] == pattern[i]:
                j += 1
            failure[i] = j
            
        # Find matches
        results = []
        j = 0
        
        for i in range(len(text)):
            while j > 0 and pattern[j] != text[i]:
                j = failure[j - 1]
            if pattern[j] == text[i]:
                j += 1
            if j == len(pattern):
                results.append(i - j + 1)
                j = failure[j - 1]
                
        return results
    
    @staticmethod
    def rabin_karp_string_matching(text, pattern, q=101):
        """Implementation of Rabin-Karp string matching algorithm."""
        n, m = len(text), len(pattern)
        if m > n:
            return []
            
        # Hash function: using polynomial rolling hash
        d = 256  # Number of possible characters
        
        # Calculate hash values for pattern and first window of text
        pattern_hash = 0
        text_hash = 0
        h = 1
        
        # The value of h would be "pow(d, m-1) % q"
        for i in range(m - 1):
            h = (h * d) % q
            
        for i in range(m):
            pattern_hash = (d * pattern_hash + ord(pattern[i])) % q
            text_hash = (d * text_hash + ord(text[i])) % q
            
        results = []
        
        # Slide the pattern over text one by one
        for i in range(n - m + 1):
            # Check if the hash values match
            if pattern_hash == text_hash:
                # Check characters one by one
                match = True
                for j in range(m):
                    if text[i + j] != pattern[j]:
                        match = False
                        break
                if match:
                    results.append(i)
                    
            # Calculate hash value for next window
            if i < n - m:
                text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % q
                
                # We might get negative value, convert it to positive
                if text_hash < 0:
                    text_hash += q
                    
        return results
    
    @staticmethod
    def levenshtein_distance(s1, s2):
        """Implementation of Levenshtein distance (edit distance) algorithm."""
        m, n = len(s1), len(s2)
        
        # Create a matrix of size (m+1) x (n+1)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + cost  # Substitution
                )
                
        return dp[m][n]
    
    @staticmethod
    def longest_common_subsequence(s1, s2):
        """Implementation of longest common subsequence algorithm."""
        m, n = len(s1), len(s2)
        
        # Create a matrix of size (m+1) x (n+1)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    
        # Reconstruct the LCS
        lcs = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if s1[i - 1] == s2[j - 1]:
                lcs.append(s1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
                
        return ''.join(reversed(lcs))
    
    @staticmethod
    def longest_common_substring(s1, s2):
        """Implementation of longest common substring algorithm."""
        m, n = len(s1), len(s2)
        
        # Create a matrix of size (m+1) x (n+1)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        max_length = 0
        end_pos = 0
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_pos = i
                else:
                    dp[i][j] = 0
                    
        return s1[end_pos - max_length:end_pos]


class GraphAlgorithms:
    """A collection of graph algorithms."""
    
    @staticmethod
    def breadth_first_search(graph, start_vertex):
        """Perform a breadth-first search on a graph."""
        if start_vertex not in graph.adjacency_list:
            return []
            
        result = []
        visited = set()
        queue = collections.deque([start_vertex])
        visited.add(start_vertex)
        
        while queue:
            current_vertex = queue.popleft()
            result.append(current_vertex)
            
            for neighbor, _ in graph.adjacency_list[current_vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return result
    
    @staticmethod
    def depth_first_search(graph, start_vertex):
        """Perform a depth-first search on a graph."""
        if start_vertex not in graph.adjacency_list:
            return []
            
        result = []
        visited = set()
        
        def dfs_helper(vertex):
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor, _ in graph.adjacency_list[vertex]:
                if neighbor not in visited:
                    dfs_helper(neighbor)
                    
        dfs_helper(start_vertex)
        return result
    
    @staticmethod
    def dijkstra_shortest_path(graph, start_vertex):
        """Find shortest paths from start_vertex to all other vertices using Dijkstra's algorithm."""
        if start_vertex not in graph.adjacency_list:
            return {}
            
        distances = {vertex: float('infinity') for vertex in graph.adjacency_list}
        distances[start_vertex] = 0
        previous = {vertex: None for vertex in graph.adjacency_list}
        unvisited = list(graph.adjacency_list.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda vertex: distances[vertex])
            
            if distances[current] == float('infinity'):
                break
                
            unvisited.remove(current)
            
            for neighbor, weight in graph.adjacency_list[current]:
                distance = distances[current] + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    
        return distances, previous
    
    @staticmethod
    def reconstruct_path(previous, start_vertex, end_vertex):
        """Reconstruct path from start_vertex to end_vertex using the previous dictionary."""
        path = []
        current = end_vertex
        
        while current != start_vertex:
            path.append(current)
            current = previous[current]
            if current is None:
                return []  # No path exists
                
        path.append(start_vertex)
        return list(reversed(path))
        
    @staticmethod
    def bellman_ford(graph, start_vertex):
        """Find shortest paths from start_vertex to all other vertices using Bellman-Ford algorithm."""
        if start_vertex not in graph.adjacency_list:
            return {}
            
        distances = {vertex: float('infinity') for vertex in graph.adjacency_list}
        distances[start_vertex] = 0
        previous = {vertex: None for vertex in graph.adjacency_list}
        
        # Relax edges |V| - 1 times
        for _ in range(len(graph.adjacency_list) - 1):
            for u in graph.adjacency_list:
                for v, weight in graph.adjacency_list[u]:
                    if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        previous[v] = u
                        
        # Check for negative weight cycles
        for u in graph.adjacency_list:
            for v, weight in graph.adjacency_list[u]:
                if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                    # Negative weight cycle detected
                    return None, None
                    
        return distances, previous
    
    @staticmethod
    def floyd_warshall(graph):
        """Find shortest paths between all pairs of vertices using Floyd-Warshall algorithm."""
        vertices = graph.get_vertices()
        n = len(vertices)
        
        # Initialize distance matrix
        dist = [[float('infinity') for _ in range(n)] for _ in range(n)]
        
        # Create a mapping of vertices to indices
        vertex_to_index = {vertex: i for i, vertex in enumerate(vertices)}
        
        # Initialize distances
        for i in range(n):
            dist[i][i] = 0
            
        for u in graph.adjacency_list:
            for v, weight in graph.adjacency_list[u]:
                u_idx = vertex_to_index[u]
                v_idx = vertex_to_index[v]
                dist[u_idx][v_idx] = weight
                
        # Calculate shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] != float('infinity') and dist[k][j] != float('infinity'):
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                        
        # Convert back to vertex-based dictionary
        result = {}
        for i, u in enumerate(vertices):
            result[u] = {}
            for j, v in enumerate(vertices):
                result[u][v] = dist[i][j]
                
        return result
    
    @staticmethod
    def kruskal_mst(graph):
        """Find minimum spanning tree using Kruskal's algorithm."""
        edges = graph.get_edges()
        vertices = graph.get_vertices()
        
        # Sort edges by weight
        edges.sort(key=lambda x: x[2])
        
        # Initialize disjoint set
        disjoint_set = DisjointSet()
        for vertex in vertices:
            disjoint_set.make_set(vertex)
            
        mst = []
        
        for u, v, weight in edges:
            if not disjoint_set.is_same_set(u, v):
                mst.append((u, v, weight))
                disjoint_set.union(u, v)
                
        return mst
    
    @staticmethod
    def prim_mst(graph, start_vertex=None):
        """Find minimum spanning tree using Prim's algorithm."""
        if not graph.adjacency_list:
            return []
            
        vertices = graph.get_vertices()
        
        if start_vertex is None:
            start_vertex = vertices[0]
            
        if start_vertex not in graph.adjacency_list:
            return []
            
        mst = []
        visited = {start_vertex}
        edges = [
            (weight, start_vertex, to)
            for to, weight in graph.adjacency_list[start_vertex]
        ]
        heapq.heapify(edges)
        
        while edges and len(visited) < len(vertices):
            weight, from_vertex, to_vertex = heapq.heappop(edges)
            
            if to_vertex not in visited:
                visited.add(to_vertex)
                mst.append((from_vertex, to_vertex, weight))
                
                for neighbor, edge_weight in graph.adjacency_list[to_vertex]:
                    if neighbor not in visited:
                        heapq.heappush(edges, (edge_weight, to_vertex, neighbor))
                        
        return mst
    
    @staticmethod
    def topological_sort(graph):
        """Perform a topological sort on a directed acyclic graph."""
        result = []
        visited = set()
        temp = set()  # For cycle detection
        
        def visit(vertex):
            if vertex in temp:
                # Cycle detected
                return False
                
            if vertex in visited:
                return True
                
            temp.add(vertex)
            
            for neighbor, _ in graph.adjacency_list[vertex]:
                if not visit(neighbor):
                    return False
                    
            temp.remove(vertex)
            visited.add(vertex)
            result.insert(0, vertex)
            return True
            
        for vertex in graph.adjacency_list:
            if vertex not in visited:
                if not visit(vertex):
                    return None  # Graph has a cycle
                    
        return result
    
    @staticmethod
    def is_bipartite(graph):
        """Check if a graph is bipartite."""
        if not graph.adjacency_list:
            return True
            
        colors = {}
        
        for start_vertex in graph.adjacency_list:
            if start_vertex not in colors:
                colors[start_vertex] = 0
                queue = collections.deque([start_vertex])
                
                while queue:
                    vertex = queue.popleft()
                    
                    for neighbor, _ in graph.adjacency_list[vertex]:
                        if neighbor not in colors:
                            colors[neighbor] = 1 - colors[vertex]
                            queue.append(neighbor)
                        elif colors[neighbor] == colors[vertex]:
                            return False
                            
        return True


class DynamicProgrammingAlgorithms:
    """A collection of dynamic programming algorithms."""
    
    @staticmethod
    def fibonacci(n):
        """Calculate the nth Fibonacci number using dynamic programming."""
        if n <= 0:
            return 0
        if n == 1:
            return 1
            
        fib = [0] * (n + 1)
        fib[1] = 1
        
        for i in range(2, n + 1):
            fib[i] = fib[i - 1] + fib[i - 2]
            
        return fib[n]
    
    @staticmethod
    def coin_change(coins, amount):
        """Find the minimum number of coins needed to make a given amount."""
        dp = [float('infinity')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
                
        return dp[amount] if dp[amount] != float('infinity') else -1