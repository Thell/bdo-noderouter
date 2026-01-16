use std::cell::RefCell;
use std::rc::Rc;
use std::vec;

use smallvec::SmallVec;

const INT_MAX: usize = usize::MAX;
const MAXDEP: i32 = 32768;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LinkNode {
    pub id: i32,
    pub prev: Option<Rc<RefCell<LinkNode>>>,
    pub next: Option<Rc<RefCell<LinkNode>>>,
}

impl LinkNode {
    pub fn new() -> Self {
        LinkNode {
            id: -1,
            prev: None,
            next: None,
        }
    }

    pub fn isolate(&mut self) {
        // UnlinkDS disconnects the vertex 𝑢 from its parent. It
        // removes 𝑢 from the children DLL of its parent.
        // For ease of presentation, we add virtual vertices
        // at the beginning and end of DLL.
        // Where the paper states UnlinkDS(u) where u is a DSNode here
        // we use self on the LinkNode itself.
        let tmp = self.prev.clone();
        if self.prev.is_some() {
            self.prev.as_ref().unwrap().borrow_mut().next = self.next.clone();
            self.prev = None;
        }
        if self.next.is_some() {
            self.next.as_ref().unwrap().borrow_mut().prev = tmp;
            self.next = None;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Node {
    // For ID-Tree spanning trees
    parent: i32,
    subtree_size: i32,

    // For disjoint sets
    root: i32,
    children_start: Rc<RefCell<LinkNode>>,
    children_end: Rc<RefCell<LinkNode>>,

    // Neighbors is only guaranteed after flush processes the edge_buf entries.
    neighbors: SmallVec<[usize; 4]>,

    // For temporal (ordered) operations
    delete_edge_buf: Vec<(i32, usize)>,
    insert_edge_buf: Vec<(i32, usize)>,
}

impl Node {
    fn new() -> Self {
        let node = Node {
            parent: -1,
            subtree_size: 0,
            root: -1,
            children_start: Rc::new(RefCell::new(LinkNode::new())),
            children_end: Rc::new(RefCell::new(LinkNode::new())),
            neighbors: SmallVec::new(),
            delete_edge_buf: vec![],
            insert_edge_buf: vec![],
        };
        node.children_start.borrow_mut().next = Some(node.children_end.clone());
        node.children_end.borrow_mut().prev = Some(node.children_start.clone());
        node
    }

    fn insert_neighbor(&mut self, u: i32) {
        let index = self.insert_edge_buf.len() + self.delete_edge_buf.len();
        self.insert_edge_buf.push((u, index));
    }

    fn insert_l_node(&mut self, v: Rc<RefCell<LinkNode>>) {
        // The paper states:
        //  7 Procedure LinkDS(𝑢,𝑣)
        //    /* union without find and comparing size */
        //    /* the input satisfies 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑢) ≤ 𝑠𝑡_𝑠𝑖𝑧𝑒(𝑣) */
        //    /* union twoDS-Trees */
        //  8 DSnode(𝑢).𝑝𝑎𝑟𝑒𝑛𝑡 ← DSnode(𝑣);
        //    /* add 𝑢 to the new DLL */
        //  9 DSnode(𝑢).𝑝𝑟𝑒 ← DSnode(𝑣).𝑐ℎ𝑖𝑙𝑑𝑟𝑒𝑛;
        // 10 DSnode(𝑢).𝑛𝑒𝑥𝑡 ← DSnode(𝑣).𝑐ℎ𝑖𝑙𝑑𝑟𝑒𝑛.𝑛𝑒𝑥𝑡;
        // 11 DSnode(𝑢).𝑝𝑟𝑒.𝑛𝑒𝑥𝑡 ← DSnode(𝑢);
        // 12 DSnode(𝑢).𝑛𝑒𝑥𝑡.𝑝𝑟𝑒 ← DSnode(𝑢);
        // If self is DSnode(𝑣) and v is DSnode(𝑢) then
        // to implement this the way the paper states then it would be:
        v.borrow_mut().prev = Some(self.children_start.clone());
        v.borrow_mut().next = self.children_start.borrow().next.clone();
        v.borrow_mut().prev.as_ref().unwrap().borrow_mut().next = Some(v.clone());
        v.borrow_mut().next.as_ref().unwrap().borrow_mut().prev = Some(v.clone());

        // but then we can't set DSnode(𝑢).𝑝𝑎𝑟𝑒𝑛𝑡 here because Node has the parent attribute, not
        // LinkNode which is what we have from v. So if we mentally swap the u and v then we'd
        // have this which isn't even possible because v is a LinkNode, not a Node...
        //
        // self.parent = v.borrow().v;
        // self.l_start.borrow_mut().prev = v.l_start.clone();
        // self.l_start.borrow_mut().prev = v.l_start.clone();
        // self.l_start.borrow_mut().next.as_ref().unwrap().borrow_mut().prev = Some(v.clone());
        // self.l_start.borrow_mut().prev.as_ref().unwrap().borrow_mut().next = Some(v.clone());
        //
        // so perhaps the actual issue with the translation -vs- the paper is the paper expects
        // 2 DSNode pointers and the C++ reference code opted to pass a LinkNode instead...

        // The C++ reference code changes both v and u...
        // v.borrow_mut().prev = Some(self.l_start.clone());
        // v.borrow_mut().next = self.l_start.borrow().next.clone();
        // self.l_start.borrow_mut().next.as_ref().unwrap().borrow_mut().prev = Some(v.clone());
        // self.l_start.borrow_mut().next = Some(v.clone());

        // Both the current and the translated C++ give the same results...
    }

    fn flush(&mut self) {
        if self.insert_edge_buf.is_empty() && self.delete_edge_buf.is_empty() {
            return;
        }

        self.insert_edge_buf.sort();
        self.delete_edge_buf.sort();

        let mut i = 0;
        let mut d = 0;
        let num_inserts = self.insert_edge_buf.len();
        let num_deletes = self.delete_edge_buf.len();

        let mut updated_neighbors = SmallVec::with_capacity(self.neighbors.len());

        for j in 0..=self.neighbors.len() {
            let v = if j < self.neighbors.len() {
                self.neighbors[j]
            } else {
                INT_MAX
            };

            while i < num_inserts && self.insert_edge_buf[i].0 < v as i32 {
                while i + 1 < num_inserts
                    && self.insert_edge_buf[i].0 == self.insert_edge_buf[i + 1].0
                {
                    i += 1;
                }
                while d < num_deletes && self.delete_edge_buf[d].0 < self.insert_edge_buf[i].0 {
                    d += 1;
                }
                while d < num_deletes && self.delete_edge_buf[d].0 == self.insert_edge_buf[i].0 {
                    d += 1;
                }
                if d > 0
                    && self.delete_edge_buf[d - 1].0 == self.insert_edge_buf[i].0
                    && self.delete_edge_buf[d - 1].1 > self.insert_edge_buf[i].1
                {
                    i += 1;
                    continue;
                }
                updated_neighbors.push(self.insert_edge_buf[i].0 as usize);
                i += 1;
            }

            if j == self.neighbors.len() {
                break;
            }

            while d < num_deletes && self.delete_edge_buf[d].0 < v as i32 {
                d += 1;
            }

            if d >= num_deletes || self.delete_edge_buf[d].0 > v as i32 {
                updated_neighbors.push(v);
                while i < num_inserts && self.insert_edge_buf[i].0 == v as i32 {
                    i += 1;
                }
                continue;
            }

            if i < num_inserts && self.insert_edge_buf[i].0 == v as i32 {
                let mut insert_stamp = 0;
                while i < num_inserts && self.insert_edge_buf[i].0 == v as i32 {
                    insert_stamp = self.insert_edge_buf[i].1;
                    i += 1;
                }

                let mut delete_stamp = 0;
                while d < num_deletes && self.delete_edge_buf[d].0 == v as i32 {
                    delete_stamp = self.delete_edge_buf[d].1;
                    d += 1;
                }

                if insert_stamp > delete_stamp {
                    updated_neighbors.push(v);
                }
            }
        }

        self.insert_edge_buf.clear();
        self.delete_edge_buf.clear();
        self.neighbors = updated_neighbors;
    }
}

#[derive(Clone, Debug)]
pub struct DSTree {
    n: usize,
    nodes: Vec<Node>,
    l_nodes: Vec<Rc<RefCell<LinkNode>>>,

    used: Vec<bool>,
    q: Vec<i32>,
    use_union_find: bool,
}

impl DSTree {
    pub fn new(adj_dict: &[SmallVec<[usize; 4]>], use_union_find: bool) -> Self {
        let mut instance = Self::setup(adj_dict, use_union_find);
        instance.initialize();
        instance
    }

    // MARK: Core
    pub fn insert_edge(&mut self, u: usize, v: usize) -> i32 {
        if !self.insert_edge_in_graph(u, v) {
            return -1;
        }
        self.insert_edge_balanced(u, v)
    }

    pub fn query(&mut self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n {
            return false;
        }

        let mut root_u = u;
        while self.nodes[root_u].parent != -1 {
            root_u = self.nodes[root_u].parent as usize;
        }

        let mut root_v = v;
        while self.nodes[root_v].parent != -1 {
            root_v = self.nodes[root_v].parent as usize;
        }

        root_u == root_v
    }

    pub fn reset_all_edges(&mut self) {
        for i in 0..self.n {
            self.nodes[i].insert_edge_buf.clear();
            self.nodes[i].delete_edge_buf.clear();
            self.nodes[i].neighbors.clear();
            self.nodes[i].parent = -1;
            self.nodes[i].subtree_size = 1;

            if self.use_union_find {
                self.nodes[i].root = i as i32;

                self.nodes[i].children_start.borrow_mut().next =
                    Some(self.nodes[i].children_end.clone());
                self.nodes[i].children_end.borrow_mut().prev =
                    Some(self.nodes[i].children_start.clone());
                self.nodes[i].children_start.borrow_mut().prev = None;
                self.nodes[i].children_end.borrow_mut().next = None;

                self.l_nodes[i].borrow_mut().prev = None;
                self.l_nodes[i].borrow_mut().next = None;

                self.nodes[i].insert_l_node(self.l_nodes[i].clone());
            }
        }
    }
}

impl DSTree {
    fn setup(adj_dict: &[SmallVec<[usize; 4]>], use_union_find: bool) -> Self {
        let n = adj_dict.len();

        let mut nodes: Vec<Node> = Vec::with_capacity(n);
        for _ in 0..n {
            nodes.push(Node::new());
        }
        for (i, node) in nodes.iter_mut().enumerate().take(n) {
            node.neighbors = adj_dict.get(i).unwrap_or(&SmallVec::new()).clone();
        }

        Self {
            n,
            nodes,
            l_nodes: vec![],
            used: vec![],
            q: vec![],
            use_union_find,
        }
    }

    fn initialize(&mut self) {
        let n = self.n;
        let use_union_find = self.use_union_find;

        let mut s: Vec<(i32, i32)> = vec![];
        self.used = vec![false; n];
        for i in 0..n {
            self.nodes[i].flush();
            let length = self.nodes[i].neighbors.len() as i32;
            s.push((length, -(i as i32)));
        }
        s.sort();

        if use_union_find {
            self.l_nodes = (0..n)
                .map(|_| Rc::new(RefCell::new(LinkNode::new())))
                .collect();
            for v in 0..n {
                self.l_nodes[v].borrow_mut().id = v as i32;
                self.l_nodes[v].borrow_mut().prev = None;
                self.l_nodes[v].borrow_mut().next = None;
                self.nodes[v].root = v as i32;
                self.nodes[v].children_start.borrow_mut().next =
                    Some(self.nodes[v].children_end.clone());
                self.nodes[v].children_end.borrow_mut().prev =
                    Some(self.nodes[v].children_start.clone());
                self.nodes[v].children_start.borrow_mut().prev = None;
                self.nodes[v].children_end.borrow_mut().next = None;
            }
        }

        for v in 0..n {
            self.nodes[v].parent = -1;
            self.nodes[v].subtree_size = 1;
        }

        for i in (0..n).rev() {
            let f = (-s[i].1) as usize;
            if self.used[f] {
                continue;
            }
            self.q.clear();
            self.used[f] = true;
            self.q.push(f as i32);

            if use_union_find {
                self.nodes[f].insert_l_node(self.l_nodes[f].clone());
            }

            let mut s_index = 0;
            while s_index < self.q.len() {
                let p = self.q[s_index];
                for j in 0..self.nodes[p as usize].neighbors.len() {
                    let v = self.nodes[p as usize].neighbors[j];
                    if !self.used[v] {
                        self.used[v] = true;
                        self.q.push(v as i32);
                        self.nodes[v].parent = p;
                        if use_union_find {
                            self.nodes[v].root = f as i32;
                            self.nodes[f].insert_l_node(self.l_nodes[v].clone());
                        }
                    }
                }
                s_index += 1;
            }

            let mut i = self.q.len() - 1;
            while i > 0 {
                let q_idx = self.q[i] as usize;
                let p_idx = self.nodes[q_idx].parent as usize;
                self.nodes[p_idx].subtree_size += self.nodes[q_idx].subtree_size;
                i -= 1;
            }

            let mut r: i32 = -1;
            let ss = self.q.len() / 2;
            for i in (0..self.q.len()).rev() {
                if r == -1 && self.nodes[self.q[i] as usize].subtree_size as usize > ss {
                    r = self.q[i];
                }
            }
            if r != f as i32 {
                self.reroot(r as usize, f as i32);
            }
        }
        self.used.fill(false);
    }

    fn find(&mut self, u: usize) -> usize {
        if self.nodes[u].root != u as i32 {
            let root = self.find(self.nodes[u].root as usize);
            if self.nodes[u].root != root as i32 {
                self.nodes[u].root = root as i32;
                self.l_nodes[u].borrow_mut().isolate();
                self.nodes[root].insert_l_node(self.l_nodes[u].clone());
            }
        }
        self.nodes[u].root as usize
    }

    fn insert_edge_in_graph(&mut self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n || u == v {
            return false;
        }
        self.nodes[u].insert_neighbor(v as i32);
        self.nodes[v].insert_neighbor(u as i32);
        true
    }

    fn insert_edge_balanced(&mut self, mut u: usize, mut v: usize) -> i32 {
        // DND-Insert
        let (mut root_u, mut root_v, mut parent_u, mut parent_v);

        // Find the roots of u and v
        if !self.use_union_find {
            root_u = u;
            while self.nodes[root_u].parent != -1 {
                root_u = self.nodes[root_u].parent as usize;
            }
            root_v = v;
            while self.nodes[root_v].parent != -1 {
                root_v = self.nodes[root_v].parent as usize;
            }
        } else {
            root_u = self.find(u);
            root_v = self.find(v);
        }

        // ID-Tree insert
        if root_u == root_v {
            let mut reshape = false;
            let mut depth = 0;
            parent_u = self.nodes[u].parent;
            parent_v = self.nodes[v].parent;
            while depth < MAXDEP {
                if parent_u == -1 {
                    if parent_v != -1 && self.nodes[parent_v as usize].parent == -1 {
                        reshape = true;
                        std::mem::swap(&mut u, &mut v);
                        std::mem::swap(&mut parent_u, &mut parent_v);
                    }
                    break;
                } else if parent_v == -1 {
                    if parent_u == -1 && self.nodes[parent_u as usize].parent == -1 {
                        reshape = true;
                    }
                    break;
                }
                parent_u = self.nodes[parent_u as usize].parent;
                parent_v = self.nodes[parent_v as usize].parent;
                depth += 1;
            }

            if reshape {
                let mut dlt = 0;
                while parent_u != -1 {
                    dlt += 1;
                    parent_u = self.nodes[parent_u as usize].parent;
                }

                dlt = dlt / 2 - 1;
                parent_u = u as i32;
                while dlt > 0 {
                    parent_u = self.nodes[parent_u as usize].parent;
                    dlt -= 1;
                }

                parent_v = self.nodes[parent_u as usize].parent;
                while parent_v != -1 {
                    self.nodes[parent_v as usize].subtree_size -=
                        self.nodes[parent_u as usize].subtree_size;
                    parent_v = self.nodes[parent_v as usize].parent;
                }

                self.nodes[parent_u as usize].parent = -1;
                self.reroot(u, -1);

                self.nodes[u].parent = v as i32;

                let s = (self.nodes[root_u].subtree_size + self.nodes[u].subtree_size) / 2;
                let mut new_root = -1;
                parent_u = v as i32;
                while parent_u != -1 {
                    self.nodes[parent_u as usize].subtree_size += self.nodes[u].subtree_size;
                    if new_root == -1 && self.nodes[parent_u as usize].subtree_size > s {
                        new_root = parent_u;
                    }
                    parent_u = self.nodes[parent_u as usize].parent;
                }
                if new_root != root_u as i32 {
                    self.reroot(new_root as usize, root_u as i32);
                }
            }
            return 0;
        }

        if self.nodes[root_u].subtree_size > self.nodes[root_v].subtree_size {
            std::mem::swap(&mut u, &mut v);
            std::mem::swap(&mut root_u, &mut root_v);
        }

        parent_u = self.nodes[u].parent;
        self.nodes[u].parent = v as i32;
        while parent_u != -1 {
            parent_v = self.nodes[parent_u as usize].parent;
            self.nodes[parent_u as usize].parent = u as i32;
            u = parent_u as usize;
            parent_u = parent_v;
        }

        let s = (self.nodes[root_u].subtree_size + self.nodes[root_v].subtree_size) / 2;
        let mut new_root = -1;
        parent_u = v as i32;
        while parent_u != -1 {
            self.nodes[parent_u as usize].subtree_size += self.nodes[root_u].subtree_size;
            if new_root == -1 && self.nodes[parent_u as usize].subtree_size > s {
                new_root = parent_u;
            }
            parent_u = self.nodes[parent_u as usize].parent;
        }

        parent_u = self.nodes[u].parent;
        while parent_u != v as i32 {
            self.nodes[u].subtree_size -= self.nodes[parent_u as usize].subtree_size;
            self.nodes[parent_u as usize].subtree_size += self.nodes[u].subtree_size;
            u = parent_u as usize;
            parent_u = self.nodes[u].parent;
        }

        if self.use_union_find {
            self.union(root_u, root_v);
        }

        if new_root != root_v as i32 {
            self.reroot(new_root as usize, root_v as i32);
        }

        1
    }

    fn reroot(&mut self, mut u: usize, f: i32) {
        // Rotate the tree containing u, making u the new root.
        // Updates the parent-child relationship from 𝑢 to the original root.
        let mut p = self.nodes[u].parent;
        let mut pp;
        self.nodes[u].parent = -1;
        while p != -1 {
            pp = self.nodes[p as usize].parent;
            self.nodes[p as usize].parent = u as i32;
            u = p as usize;
            p = pp;
        }
        // Update the subtree size attribute from 𝑢 to the original root.
        p = self.nodes[u].parent;
        while p != -1 {
            self.nodes[u].subtree_size -= self.nodes[p as usize].subtree_size;
            self.nodes[p as usize].subtree_size += self.nodes[u].subtree_size;
            u = p as usize;
            p = self.nodes[p as usize].parent;
        }

        //
        if self.use_union_find && f >= 0 {
            self.nodes[f as usize].root = u as i32;
            self.l_nodes[f as usize].borrow_mut().isolate();
            self.nodes[u].insert_l_node(self.l_nodes[f as usize].clone());

            self.nodes[u].root = u as i32;
            self.l_nodes[u].borrow_mut().isolate();
            self.nodes[u].insert_l_node(self.l_nodes[u].clone());
        }
    }

    fn union(&mut self, root_u: usize, root_v: usize) {
        if root_u == root_v {
            return;
        }
        self.nodes[root_u].root = root_v as i32;
        self.l_nodes[root_u].borrow_mut().isolate();
        self.nodes[root_v].insert_l_node(self.l_nodes[root_u].clone());
    }
}
