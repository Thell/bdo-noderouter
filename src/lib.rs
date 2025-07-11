use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use std::cell::RefCell;
use std::rc::Rc;
use std::vec;

const INT_MAX: i32 = i32::MAX;
const MAXDEP: i32 = 32768;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LinkNode {
    pub v: i32,
    pub prev: Option<Rc<RefCell<LinkNode>>>,
    pub next: Option<Rc<RefCell<LinkNode>>>,
}

impl LinkNode {
    pub fn new() -> Self {
        LinkNode {
            v: -1,
            prev: None,
            next: None,
        }
    }

    pub fn isolate(&mut self) {
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
    // for graph
    adj: Vec<i32>,

    // for tree
    p: i32,       // parent node in the tree
    sub_cnt: i32, // number of descendants in the tree

    // for union_find
    f: i32,
    l_start: Rc<RefCell<LinkNode>>,
    l_end: Rc<RefCell<LinkNode>>,

    // for buffered adj operations
    del_buf: Vec<(i32, usize)>,
    ins_buf: Vec<(i32, usize)>,
}

impl Node {
    fn new() -> Self {
        let node = Node {
            adj: vec![],
            p: -1,
            sub_cnt: 0,
            f: -1,
            l_start: Rc::new(RefCell::new(LinkNode::new())),
            l_end: Rc::new(RefCell::new(LinkNode::new())),
            del_buf: vec![],
            ins_buf: vec![],
        };
        node.l_start.borrow_mut().next = Some(node.l_end.clone());
        node.l_end.borrow_mut().prev = Some(node.l_start.clone());
        node
    }

    fn insert_adj(&mut self, u: i32) {
        let index = self.ins_buf.len() + self.del_buf.len();
        self.ins_buf.push((u, index));
    }

    fn delete_adj(&mut self, u: i32) {
        let index = self.ins_buf.len() + self.del_buf.len();
        self.del_buf.push((u, index));
    }

    fn insert_l_node(&mut self, v: Rc<RefCell<LinkNode>>) {
        v.borrow_mut().next = self.l_start.borrow().next.clone();
        v.borrow_mut().prev = Some(self.l_start.clone());
        self.l_start
            .borrow_mut()
            .next
            .as_ref()
            .unwrap()
            .borrow_mut()
            .prev = Some(v.clone());
        self.l_start.borrow_mut().next = Some(v.clone());
    }

    fn insert_l_nodes(&mut self, v: &Node) {
        let v_next_opt = v.l_start.borrow().next.clone();
        if v_next_opt
            .as_ref()
            .map_or(true, |n| Rc::ptr_eq(n, &v.l_end))
            || std::ptr::eq(self, v)
        {
            return;
        }

        let s = v_next_opt.unwrap();
        let t = v.l_end.borrow().prev.as_ref().unwrap().clone();

        t.borrow_mut().next = self.l_start.borrow().next.clone();
        if let Some(next) = self.l_start.borrow().next.clone() {
            next.borrow_mut().prev = Some(t.clone());
        }

        s.borrow_mut().prev = Some(self.l_start.clone());
        self.l_start.borrow_mut().next = Some(s);

        v.l_start.borrow_mut().next = Some(v.l_end.clone());
        v.l_end.borrow_mut().prev = Some(v.l_start.clone());
    }

    fn flush(&mut self) {
        if self.ins_buf.is_empty() && self.del_buf.is_empty() {
            return;
        }

        self.ins_buf.sort();
        self.del_buf.sort();

        let mut i = 0;
        let mut d = 0;
        let ni = self.ins_buf.len();
        let nd = self.del_buf.len();

        let mut l = vec![];

        for j in 0..=self.adj.len() {
            let v = if j < self.adj.len() {
                self.adj[j]
            } else {
                INT_MAX
            };

            while i < ni && self.ins_buf[i].0 < v {
                while i + 1 < ni && self.ins_buf[i].0 == self.ins_buf[i + 1].0 {
                    i += 1;
                }
                while d < nd && self.del_buf[d].0 < self.ins_buf[i].0 {
                    d += 1;
                }
                while d < nd && self.del_buf[d].0 == self.ins_buf[i].0 {
                    d += 1;
                }
                if d > 0
                    && self.del_buf[d - 1].0 == self.ins_buf[i].0
                    && self.del_buf[d - 1].1 > self.ins_buf[i].1
                {
                    i += 1;
                    continue;
                }
                l.push(self.ins_buf[i].0);
                i += 1;
            }

            if j == self.adj.len() {
                break;
            }

            while d < nd && self.del_buf[d].0 < v {
                d += 1;
            }

            if d >= nd || self.del_buf[d].0 > v {
                l.push(v);
                while i < ni && self.ins_buf[i].0 == v {
                    i += 1;
                }
                continue;
            }

            if i < ni && self.ins_buf[i].0 == v {
                let mut ti = 0;
                while i < ni && self.ins_buf[i].0 == v {
                    ti = self.ins_buf[i].1;
                    i += 1;
                }

                let mut td = 0;
                while d < nd && self.del_buf[d].0 == v {
                    td = self.del_buf[d].1;
                    d += 1;
                }

                if ti > td {
                    l.push(v);
                }
            }
        }

        self.ins_buf.clear();
        self.del_buf.clear();
        self.adj = l;
    }
}

#[derive(Clone, Debug)]
#[pyclass(unsendable)]
struct DynamicCC {
    n: usize,
    nodes: Vec<Node>,
    l_nodes: Vec<Rc<RefCell<LinkNode>>>,

    used: Vec<bool>,
    q: Vec<i32>,
    l: Vec<usize>,

    use_union_find: bool,
}

#[pymethods]
impl DynamicCC {
    #[new]
    fn new(adj_dict: HashMap<i32, Vec<i32>>, use_union_find: bool) -> Self {
        let mut instance = Self::setup(&adj_dict, use_union_find);
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

    pub fn delete_edge(&mut self, u: usize, v: usize) -> i32 {
        if !self.delete_edge_in_graph(u, v) {
            return -1;
        }
        self.delete_edge_balanced(u, v)
    }

    pub fn query(&mut self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n {
            return false;
        }

        let mut root_u = u;
        while self.nodes[root_u].p != -1 {
            root_u = self.nodes[root_u].p as usize;
        }

        let mut root_v = v;
        while self.nodes[root_v].p != -1 {
            root_v = self.nodes[root_v].p as usize;
        }

        root_u == root_v
    }

    // MARK: Extensions
    pub fn cycle_basis(&mut self, root: Option<usize>) -> Vec<Vec<usize>> {
        use std::collections::{HashMap, HashSet};

        if let Some(r) = root {
            if r >= self.n {
                return vec![];
            }
        }

        let nodes_to_use: Vec<usize> = match root {
            Some(r) => self.node_connected_component(r),
            None => (0..self.n).collect(),
        };

        for v in nodes_to_use.iter() {
            self.nodes[*v].flush();
        }

        let mut gnodes: HashMap<usize, ()> = nodes_to_use.iter().map(|&v| (v, ())).collect();
        let mut cycles: Vec<Vec<usize>> = vec![];
        let mut current_root = root;

        while !gnodes.is_empty() {
            if current_root.is_none() {
                current_root = gnodes.keys().next().cloned();
            }
            let r = current_root.unwrap();
            gnodes.remove(&r);

            let mut stack = vec![r];
            let mut pred: HashMap<usize, usize> = HashMap::new();
            let mut used: HashMap<usize, HashSet<usize>> = HashMap::new();

            pred.insert(r, r);
            used.insert(r, HashSet::new());

            while let Some(z) = stack.pop() {
                let zused = used.get(&z).cloned().unwrap_or_default();
                let neighbors: Vec<usize> = self.nodes[z].adj.iter().map(|&n| n as usize).collect();

                for &neighbor in neighbors.iter().rev() {
                    if !used.contains_key(&neighbor) {
                        pred.insert(neighbor, z);
                        stack.push(neighbor);
                        used.insert(neighbor, HashSet::from([z]));
                    } else if neighbor == z {
                        cycles.push(vec![z]);
                    } else if !zused.contains(&neighbor) {
                        let pn = used.get(&neighbor).cloned().unwrap_or_default();
                        let mut cycle = vec![neighbor, z];
                        let mut p = pred[&z];
                        while !pn.contains(&p) {
                            cycle.push(p);
                            p = pred[&p];
                        }
                        cycle.push(p);
                        cycles.push(cycle);
                        used.get_mut(&neighbor).unwrap().insert(z);
                    }
                }
            }

            for node in pred.keys() {
                gnodes.remove(node);
            }

            current_root = None;
        }

        cycles
    }

    pub fn node_connected_component(&mut self, v: usize) -> Vec<usize> {
        if v >= self.n {
            return vec![];
        }

        let mut component_nodes = vec![v];
        let mut stack = vec![v];
        let mut visited = HashSet::new();
        visited.insert(v);

        self.nodes[v].flush();

        while let Some(node) = stack.pop() {
            let neighbors: Vec<usize> = self.nodes[node].adj.iter().map(|&n| n as usize).collect();
            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    stack.push(neighbor);
                    component_nodes.push(neighbor);
                    self.nodes[neighbor].flush();
                }
            }
        }

        component_nodes
    }

    pub fn num_connected_components(&mut self) -> usize {
        let mut num_components = 0;
        let mut visited = HashSet::new();

        for v in 0..self.n {
            if !visited.contains(&v) {
                let comp = self.node_connected_component(v);
                if comp.len() > 1 {
                    num_components += 1;
                }
                for &node in &comp {
                    visited.insert(node);
                }
            }
        }

        num_components
    }

    pub fn active_node_indices(&mut self) -> Vec<usize> {
        (0..self.n)
            .filter(|&i| !self.is_isolated(i, Some(true)))
            .collect()
    }

    pub fn isolate_node(&mut self, v: usize) {
        if v >= self.n {
            return;
        }

        self.nodes[v].flush();

        let neighbors = self.nodes[v].adj.clone();
        for &neighbor in &neighbors {
            self.delete_edge(v, neighbor as usize);
        }

        self.nodes[v].flush();
    }

    pub fn isolate_nodes(&mut self, nodes: Vec<usize>) {
        for v in nodes {
            self.isolate_node(v);
        }
    }

    pub fn is_isolated(&mut self, v: usize, flush: Option<bool>) -> bool {
        if v >= self.n {
            return false;
        }

        if flush.unwrap_or(true) {
            self.nodes[v].flush();
        }

        self.nodes[v].adj.is_empty()
    }

    pub fn degree(&mut self, v: usize) -> i32 {
        if v >= self.n {
            return -1;
        }

        self.nodes[v].flush();
        self.nodes[v].adj.len() as i32
    }

    pub fn neighbors(&mut self, v: usize) -> Vec<i32> {
        if v >= self.n {
            return vec![];
        }

        self.nodes[v].flush();
        self.nodes[v].adj.clone()
    }

    pub fn potential_neighbors_from(
        &mut self,
        v: usize,
        from_indices: Vec<usize>,
        flush: Option<bool>,
    ) -> Vec<usize> {
        if v >= self.n {
            return vec![];
        }

        if flush.unwrap_or(true) {
            self.nodes[v].flush();
        }

        from_indices
            .into_iter()
            .filter(|&neighbor| !self.is_isolated(neighbor, flush))
            .collect()
    }

    fn get_f(&mut self, u: usize) -> usize {
        if self.nodes[u].f != u as i32 {
            let f = self.get_f(self.nodes[u].f as usize);
            if self.nodes[u].f != f as i32 {
                self.nodes[u].f = f as i32;
                self.l_nodes[u].borrow_mut().isolate();
                self.nodes[f].insert_l_node(self.l_nodes[u].clone());
            }
        }
        self.nodes[u].f as usize
    }
}

impl DynamicCC {
    fn setup(adj_dict: &HashMap<i32, Vec<i32>>, use_union_find: bool) -> Self {
        let n = adj_dict.len();

        let mut nodes: Vec<Node> = Vec::with_capacity(n);
        for _ in 0..n {
            nodes.push(Node::new());
        }
        for i in 0..n {
            nodes[i].adj = adj_dict.get(&(i as i32)).unwrap_or(&vec![]).clone();
        }

        Self {
            n,
            nodes,
            l_nodes: vec![],
            used: vec![],
            q: vec![],
            l: vec![],
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
            let length = self.nodes[i].adj.len() as i32;
            s.push((length, -(i as i32)));
        }
        s.sort();

        if use_union_find {
            self.l_nodes = (0..n)
                .map(|_| Rc::new(RefCell::new(LinkNode::new())))
                .collect();
            for v in 0..n {
                self.l_nodes[v].borrow_mut().v = v as i32;
                self.l_nodes[v].borrow_mut().prev = None;
                self.l_nodes[v].borrow_mut().next = None;
                self.nodes[v].f = v as i32;
                self.nodes[v].l_start.borrow_mut().next = Some(self.nodes[v].l_end.clone());
                self.nodes[v].l_end.borrow_mut().prev = Some(self.nodes[v].l_start.clone());
                self.nodes[v].l_start.borrow_mut().prev = None;
                self.nodes[v].l_end.borrow_mut().next = None;
            }
        }

        for v in 0..n {
            self.nodes[v].p = -1;
            self.nodes[v].sub_cnt = 1;
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
                for j in 0..self.nodes[p as usize].adj.len() {
                    let v = self.nodes[p as usize].adj[j] as usize;
                    if !self.used[v] {
                        self.used[v] = true;
                        self.q.push(v as i32);
                        self.nodes[v].p = p as i32;
                        if use_union_find {
                            self.nodes[v].f = f as i32;
                            self.nodes[f].insert_l_node(self.l_nodes[v].clone());
                        }
                    }
                }
                s_index += 1;
            }

            let mut i = self.q.len() - 1;
            while i > 0 {
                let q_idx = self.q[i as usize] as usize;
                let p_idx = self.nodes[q_idx].p as usize;
                self.nodes[p_idx].sub_cnt += self.nodes[q_idx].sub_cnt;
                i -= 1;
            }

            let mut r: i32 = -1;
            let ss = self.q.len() / 2;
            for i in (0..self.q.len()).rev() {
                if r == -1 && self.nodes[self.q[i] as usize].sub_cnt as usize > ss {
                    r = self.q[i] as i32;
                }
            }
            if r != f as i32 {
                self.reroot(r as usize, f as i32);
            }
        }
        self.used.fill(false);
    }

    fn insert_edge_in_graph(&mut self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n || u == v {
            return false;
        }
        self.nodes[u].insert_adj(v as i32);
        self.nodes[v].insert_adj(u as i32);
        true
    }

    fn insert_edge_balanced(&mut self, mut u: usize, mut v: usize) -> i32 {
        let (mut fu, mut fv, mut p, mut pp);
        if !self.use_union_find {
            fu = u;
            while self.nodes[fu].p != -1 {
                fu = self.nodes[fu].p as usize;
            }
            fv = v;
            while self.nodes[fv].p != -1 {
                fv = self.nodes[fv].p as usize;
            }
        } else {
            fu = self.get_f(u);
            fv = self.get_f(v);
        }

        if fu == fv {
            let mut reshape = false;
            let mut d = 0;
            p = self.nodes[u].p;
            pp = self.nodes[v].p;
            while d < MAXDEP {
                if p == -1 {
                    if pp != -1 && self.nodes[pp as usize].p == -1 {
                        reshape = true;
                        std::mem::swap(&mut u, &mut v);
                        std::mem::swap(&mut p, &mut pp);
                    }
                    break;
                } else if pp == -1 {
                    if p == -1 && self.nodes[p as usize].p == -1 {
                        reshape = true;
                    }
                    break;
                }
                p = self.nodes[p as usize].p;
                pp = self.nodes[pp as usize].p;
                d += 1;
            }

            if reshape {
                let mut dlt = 0;
                while p != -1 {
                    dlt += 1;
                    p = self.nodes[p as usize].p;
                }

                dlt = dlt / 2 - 1;
                p = u as i32;
                while dlt > 0 {
                    p = self.nodes[p as usize].p;
                    dlt -= 1;
                }

                pp = self.nodes[p as usize].p;
                while pp != -1 {
                    self.nodes[pp as usize].sub_cnt -= self.nodes[p as usize].sub_cnt;
                    pp = self.nodes[pp as usize].p;
                }

                self.nodes[p as usize].p = -1;
                self.reroot(u, -1);

                self.nodes[u].p = v as i32;

                let s = (self.nodes[fu].sub_cnt + self.nodes[u].sub_cnt) / 2;
                let mut r = -1;
                p = v as i32;
                while p != -1 {
                    self.nodes[p as usize].sub_cnt += self.nodes[u].sub_cnt;
                    if r == -1 && self.nodes[p as usize].sub_cnt > s {
                        r = p;
                    }
                    p = self.nodes[p as usize].p;
                }
                if r != fu as i32 {
                    self.reroot(r as usize, fu as i32);
                }
            }
            return 0;
        }

        if self.nodes[fu].sub_cnt > self.nodes[fv].sub_cnt {
            std::mem::swap(&mut u, &mut v);
            std::mem::swap(&mut fu, &mut fv);
        }

        p = self.nodes[u].p;
        self.nodes[u].p = v as i32;
        while p != -1 {
            pp = self.nodes[p as usize].p;
            self.nodes[p as usize].p = u as i32;
            u = p as usize;
            p = pp;
        }

        let s = (self.nodes[fu].sub_cnt + self.nodes[fv].sub_cnt) / 2;
        let mut r = -1;
        p = v as i32;
        while p != -1 {
            self.nodes[p as usize].sub_cnt += self.nodes[fu].sub_cnt;
            if r == -1 && self.nodes[p as usize].sub_cnt > s {
                r = p;
            }
            p = self.nodes[p as usize].p;
        }

        p = self.nodes[u].p;
        while p != v as i32 {
            self.nodes[u].sub_cnt -= self.nodes[p as usize].sub_cnt;
            self.nodes[p as usize].sub_cnt += self.nodes[u].sub_cnt;
            u = p as usize;
            p = self.nodes[u].p;
        }

        if self.use_union_find {
            self.union_f(fu, fv);
        }

        if r != fv as i32 {
            self.reroot(r as usize, fv as i32);
        }

        1
    }

    fn delete_edge_in_graph(&mut self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n || u == v {
            return false;
        }
        self.nodes[u].delete_adj(v as i32);
        self.nodes[v].delete_adj(u as i32);
        true
    }

    fn delete_edge_balanced(&mut self, mut u: usize, mut v: usize) -> i32 {
        if (self.nodes[u].p != v as i32 && self.nodes[v].p != u as i32) || u == v {
            return 0;
        }
        if self.nodes[v].p == u as i32 {
            std::mem::swap(&mut u, &mut v); // ensure u -> v
        }

        let mut f = 0;
        let mut w = v as i32;
        while w != -1 {
            self.nodes[w as usize].sub_cnt -= self.nodes[u].sub_cnt;
            f = w;
            w = self.nodes[w as usize].p;
        }

        self.nodes[u].p = -1;
        let (ns, nl, need_reroot): (usize, usize, bool) =
            if self.nodes[u].sub_cnt > self.nodes[f as usize].sub_cnt {
                (f as usize, u as usize, true)
            } else {
                (u as usize, f as usize, false)
            };
        if self.use_union_find && need_reroot {
            self.nodes[f as usize].f = u as i32;
            self.l_nodes[f as usize].borrow_mut().isolate();
            self.nodes[u].insert_l_node(self.l_nodes[f as usize].clone());

            self.nodes[u].f = u as i32;
            self.l_nodes[u as usize].borrow_mut().isolate();
            self.nodes[u as usize].insert_l_node(self.l_nodes[u as usize].clone());
        }

        if self.find_replacement(ns, nl) {
            return 1;
        }

        if self.use_union_find {
            self.remove_subtree_union_find(ns, nl, need_reroot);
        }
        2
    }

    fn find_replacement(&mut self, u: usize, f: usize) -> bool {
        self.q.clear();
        self.l.clear();

        self.q.push(u as i32);
        self.l.push(u);
        self.used[u] = true;

        let mut i = 0;
        while i < self.q.len() {
            let mut x = self.q[i];
            i += 1;

            self.nodes[x as usize].flush();

            let mut j = 0;
            while j < self.nodes[x as usize].adj.len() {
                let y = self.nodes[x as usize].adj[j];
                if y == self.nodes[x as usize].p {
                    j += 1;
                    continue;
                }

                if self.nodes[y as usize].p == x as i32 {
                    self.q.push(y);
                    if !self.used[y as usize] {
                        self.used[y as usize] = true;
                        self.l.push(y as usize);
                    }
                    j += 1;
                    continue;
                }

                // Try to build a new path from y upward
                let mut succ = true;
                let mut w = y;
                while w != -1 {
                    if self.used[w as usize] {
                        succ = false;
                        break;
                    }
                    self.used[w as usize] = true;
                    self.l.push(w as usize);

                    w = self.nodes[w as usize].p;
                }
                if !succ {
                    j += 1;
                    continue;
                }

                // Reconnect path from x to y
                let mut p = self.nodes[x as usize].p;
                self.nodes[x as usize].p = y as i32;
                while p != -1 {
                    let pp = self.nodes[p as usize].p;
                    self.nodes[p as usize].p = x;
                    x = p;
                    p = pp;
                }

                // Compute new root
                let s = (self.nodes[f].sub_cnt + self.nodes[u].sub_cnt) / 2;
                let mut r = None;

                let mut p = y as i32;
                while p != -1 {
                    self.nodes[p as usize].sub_cnt += self.nodes[u].sub_cnt;
                    if r.is_none() && self.nodes[p as usize].sub_cnt > s {
                        r = Some(p as usize);
                    }
                    p = self.nodes[p as usize].p;
                }

                // Fix subtree sizes
                let mut p = self.nodes[x as usize].p;
                while p != y as i32 {
                    self.nodes[x as usize].sub_cnt -= self.nodes[p as usize].sub_cnt;
                    self.nodes[p as usize].sub_cnt += self.nodes[x as usize].sub_cnt;
                    x = p;
                    p = self.nodes[p as usize].p;
                }

                for &k in &self.l {
                    self.used[k] = false;
                }

                if r != Some(f) {
                    self.reroot(r.unwrap(), f as i32);
                }

                return true;
            }
        }

        for &k in &self.l {
            self.used[k] = false;
        }

        false
    }

    fn reroot(&mut self, mut u: usize, f: i32) {
        // Reverse parent pointers
        let mut p = self.nodes[u].p;
        let mut pp;
        self.nodes[u].p = -1;
        while p != -1 {
            pp = self.nodes[p as usize].p;
            self.nodes[p as usize].p = u as i32;
            u = p as usize;
            p = pp;
        }
        // Adjust subtree counts
        p = self.nodes[u].p;
        while p != -1 {
            self.nodes[u].sub_cnt -= self.nodes[p as usize].sub_cnt;
            self.nodes[p as usize].sub_cnt += self.nodes[u].sub_cnt;
            u = p as usize;
            p = self.nodes[p as usize].p;
        }

        if self.use_union_find && f >= 0 {
            self.nodes[f as usize].f = u as i32;
            self.l_nodes[f as usize].borrow_mut().isolate();
            self.nodes[u].insert_l_node(self.l_nodes[f as usize].clone());

            self.nodes[u].f = u as i32;
            self.l_nodes[u as usize].borrow_mut().isolate();
            self.nodes[u].insert_l_node(self.l_nodes[u as usize].clone());
        }
    }

    fn remove_subtree_union_find(&mut self, u: usize, v: usize, _need_reroot: bool) {
        let fv = v;
        let mut i = 0;
        while i < self.q.len() {
            let x = self.q[i];

            let l_start_next = self.nodes[x as usize].l_start.borrow().next.clone();
            let l_end = self.nodes[x as usize].l_end.clone();
            if let Some(mut curr) = l_start_next {
                if !Rc::ptr_eq(&curr, &l_end) {
                    while !Rc::ptr_eq(&curr, &l_end) {
                        let y_v = curr.borrow().v as usize;
                        self.nodes[y_v].f = fv as i32;

                        let next = { curr.borrow().next.clone() };
                        curr = next.unwrap();
                    }

                    let (a, b) = if fv < x as usize {
                        let (left, right) = self.nodes.split_at_mut(x as usize);
                        (&mut left[fv], &right[0])
                    } else {
                        let (left, right) = self.nodes.split_at_mut(fv);
                        (&mut right[0], &left[x as usize])
                    };

                    a.insert_l_nodes(b);
                }
            }

            i += 1;
        }

        for i in 0..self.q.len() {
            let x = self.q[i];
            self.l_nodes[x as usize].borrow_mut().isolate();
            self.nodes[u as usize].insert_l_node(self.l_nodes[x as usize].clone());
            self.nodes[x as usize].f = u as i32;
        }
    }

    fn union_f(&mut self, fu: usize, fv: usize) {
        if fu == fv {
            return;
        }
        self.nodes[fu].f = fv as i32;
        self.l_nodes[fu].borrow_mut().isolate();
        self.nodes[fv].insert_l_node(self.l_nodes[fu].clone());
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn nwsf_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DynamicCC>()?;
    Ok(())
}
