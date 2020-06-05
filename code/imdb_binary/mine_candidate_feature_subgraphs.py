import copy
import time
import math


# Class for graph entity
class Graph:
    def __init__(self):
        self.vertices = dict()
        self.adjacency = dict()

    def __repr__(self) -> str:
        return "\nVertices:" + str(self.vertices) \
               + "\nAdjacency List: " + str(self.adjacency) + "\n"


file_name = "imdb_binary"
# loading labels of graphs from file
label = open("../../data/"+file_name+"/"+file_name+"_label.txt").readlines()
labels = []
for line in label:
    labels.append(int(line.split("\t")[0]))


# loading graphs from file
data = open("../../data/"+file_name+"/"+file_name+"_graph.txt").readlines()
Graphs = dict()
Global_graphs = []
graph = None
gi = 0
for line in data:
    if line[0] == 't':
        if graph is None:
            graph = Graph()
        else:
            if labels[gi] not in Graphs:
                Graphs[labels[gi]] = []
            Graphs[labels[gi]].append(graph)
            Global_graphs.append(graph)
            gi += 1
            graph = Graph()
    if line[0] == 'v':
        line = line.split(' ')
        graph.vertices[int(line[1])] = int(line[2])
    if line[0] == 'e':
        _, vertex1, vertex2, label = line.split(" ")
        vertex1 = int(vertex1)
        vertex2 = int(vertex2)
        label = int(label)
        if vertex1 in graph.adjacency:
            graph.adjacency[vertex1].append((vertex2, label))
        else:
            graph.adjacency[vertex1] = [(vertex2, label)]
        if vertex2 in graph.adjacency:
            graph.adjacency[vertex2].append((vertex1, label))
        else:
            graph.adjacency[vertex2] = [(vertex1, label)]

if labels[gi] not in Graphs:
    Graphs[labels[gi]] = []
Graphs[labels[gi]].append(graph)
Global_graphs.append(graph)


# gSpan: finds rightmost path of a DFS code
def RightMostPath(code):
    adj = dict()
    ur = 0
    for c in code:
        ur = max(ur, c[0])
        ur = max(ur, c[1])
        if c[1] > c[0]:
            adj[c[1]] = c[0]
    result = [ur]
    u = ur
    while u != 0:
        u = adj[u]
        result.append(u)
    return ur, list(reversed(result))


# gSpan: finds rightmost path extensions of a DFS code
def RightMostExtensions(code, graphs):
    result = dict()
    for i in range(len(graphs)):
        graph = graphs[i]
        temp_result = dict()
        if code.__len__() == 0:
            for u in graph.adjacency:
                for e in graph.adjacency[u]:
                    v, edge_label = e
                    u_label = graph.vertices[u]
                    v_label = graph.vertices[v]
                    temp_result[(0, 1, u_label, v_label, edge_label)] = 1
        else:
            isomorphisms = subgraphIsomorphisms(code, graph)
            u, R = RightMostPath(code)
            for isomorphism in isomorphisms:
                for v in R:
                    if u == v:
                        continue
                    iso_u = isomorphism[u]
                    iso_v = isomorphism[v]
                    for e in graph.adjacency[iso_u]:
                        if e[0] != iso_v:
                            continue
                        edge_label = e[1]
                        exists = False
                        for c in code:
                            if c[0] == u and c[1] == v and c[4] == edge_label:
                                exists = True
                            elif c[0] == v and c[1] == u and c[4] == edge_label:
                                exists = True
                        if not exists:
                            temp_result[(u, v, graph.vertices[iso_u], graph.vertices[iso_v], edge_label)] = 1
                ur = u
                for u in R:
                    iso_u = isomorphism[u]
                    for e in graph.adjacency[iso_u]:
                        iso_v, edge_label = e
                        if iso_v in isomorphism.values():
                            continue
                        u_label, v_label = graph.vertices[iso_u], graph.vertices[iso_v]
                        temp_result[(u, ur+1, u_label, v_label, edge_label)] = 1

        for key in temp_result:
            if key in result:
                cur = result[key]
                cur.append(i)
                result[key] = cur
            else:
                result[key] = [i]
    return result


# gSpan: finds subgraph isomorphisms from a DFS code to a graph
def subgraphIsomorphisms(code, graph):
    isomorphisms = []
    l0 = code[0][2]
    for v in graph.vertices:
        if graph.vertices[v] == l0:
            isomorphisms.append({0: v})
    for tuple in code:
        u, v, u_label, v_label, edge_label = tuple
        temp_isomorphisms = []
        for isomorphism in isomorphisms:
            if v > u:
                iso_u = isomorphism[u]
                try:
                    _ = graph.adjacency[iso_u]
                except KeyError:
                    continue
                for e in graph.adjacency[iso_u]:
                    iso_v, iso_edge_label = e
                    if iso_v not in isomorphism.values() and graph.vertices[iso_v] == v_label and edge_label == iso_edge_label:
                        new_iso = copy.deepcopy(isomorphism)
                        new_iso[v] = iso_v
                        temp_isomorphisms.append(new_iso)

            else:
                iso_u = isomorphism[u]
                iso_v = isomorphism[v]
                for e in graph.adjacency[iso_u]:
                    c_iso_v, c_iso_edge_label = e
                    if c_iso_v == iso_v and edge_label == c_iso_edge_label:
                        temp_isomorphisms.append(copy.deepcopy(isomorphism))
        isomorphisms = temp_isomorphisms
    return isomorphisms


# gSpan: builds graph from DFS code
def buildGraph(code):
    graph = Graph()
    for tuple in code:
        u, v, u_label, v_label, edge_label = tuple
        graph.vertices[u] = u_label
        graph.vertices[v] = v_label
        if u in graph.adjacency:
            graph.adjacency[u].append((v, edge_label))
        else:
            graph.adjacency[u] = [(v, edge_label)]
        if v in graph.adjacency:
            graph.adjacency[v].append((u, edge_label))
        else:
            graph.adjacency[v] = [(u, edge_label)]
    return graph


# gSpan: canonical ordering of tuples
def minTuple(tuple1, tuple2):
    u1, v1, u1_label, v1_label, edge1label = tuple1
    u2, v2, u2_label, v2_label, edge2label = tuple2
    if u1 == u2 and v1 == v2:
        if u1_label < u2_label:
            return tuple1
        elif u1_label > u2_label:
            return tuple2
        elif v1_label < v2_label:
            return tuple1
        elif v1_label > v2_label:
            return tuple2
        elif edge1label < edge2label:
            return tuple1
        return tuple2
    else:
        if u1 < v1 and u2 < v2:  # both forward edge
            if v1 < v2:
                return tuple1
            elif v1 == v2 and u1 > u2:
                return tuple1
            return tuple2
        if u1 > v1 and u2 > v2:  # both backward edge
            if u1 < u2:
                return tuple1
            elif u1 == u2 and v1 < v2:
                return tuple1
            return tuple2
        if u1 < v1 and u2 > v2:  # tuple1 forward tuple2 backward
            if v1 <= u2:
                return tuple1
            return tuple2
        if u1 > v1 and u2 < v2:  # tuple1 backward tuple2 forward
            if u1 < v2:
                return tuple1
            return tuple2


# gSpan: finds minimum tuples
def minExtension(tuples):
    result = None
    for t in tuples:
        if result is None:
            result = t
        else:
            result = minTuple(result, t)
    return result


# gSpan: checks if a DFS code is canonical
def isCannonical(code):
    graph = buildGraph(code)
    c = []
    for i in range(len(code)):
        extension = minExtension(RightMostExtensions(c, [graph]))
        if minTuple(extension, code[i]) != code[i]:
            return False
        c.append(extension)
    return True


# gSpan: recursively mines frequent subgraphs
def GSpan(code, graphs, min_sup, t, codes, log):
    code = copy.deepcopy(code)
    extentions = RightMostExtensions(code, graphs)
    for key in extentions:
        sup = len(extentions[key])
        new_code = copy.deepcopy(code)
        new_code.append(key)
        if isCannonical(new_code) and sup >= min_sup:
            codes.append((new_code, extentions[key]))
            print(new_code)
            GSpan(new_code, graphs, min_sup, t, codes, log)


min_coverage = 10
coverage = [0]*len(Global_graphs)
t = time.time()
codes_bucket = []
base_sup = 0.2

# mines frequent subgraphs in multiple phases
for i in range(5):
    Graphs = []
    b = True
    for i in range(len(Global_graphs)):
        if coverage[i] < min_coverage:
            b = False
            Graphs.append(Global_graphs[i])
    if b or math.ceil(len(Graphs) * base_sup) <= 1:
        break
    codes = []
    GSpan([], Graphs, math.ceil(len(Graphs) * base_sup), t, codes, file_name)
    b = True
    for code in codes:
        if code[0] not in codes_bucket:
            b = False
            codes_bucket.append(code[0])
            for i in code[1]:
                coverage[i] += 1
    if b:
        break
    base_sup -= 0.025


# mines smallest subgraphs
for i in range(len(Global_graphs)):
    q = [[]]
    while coverage[i] < min_coverage:
        if len(q) == 0:
            break
        try:
            extensions = RightMostExtensions(q[0], [Global_graphs[i]])
        except MemoryError:
            extensions = []
        for edge in extensions:
            code = copy.deepcopy(q[0])
            code.append(edge)
            if isCannonical(code):
                q.append(code)
                if code not in codes_bucket:
                    codes_bucket.append(code)
                coverage[i] += 1
        q = q[1:]


# outputs codes and Id of graphs that is covered by the code
out = open("cover_freq.txt", 'w')
for c in range(len(codes_bucket)):
    code = codes_bucket[c]
    out.write("#" + str(code) + "\n")
    for i in range(len(Global_graphs)):
        try:
            count = len(subgraphIsomorphisms(code, Global_graphs[i]))
        except MemoryError:
            count = 0
        if count > 0:
            out.write(str(i)+" "+str(count)+"\n")
    out.write("\n")


