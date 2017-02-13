import tensorflow as tf
import numpy as np
from IPython.display import clear_output, Image, display, HTML

def graph_as_HTML(graph_def, baseURL=''):
    # Helper functions for TF Graph visualization
    def _strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped {} bytes>".format(size).encode()
        return strip_def

    def _rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])
        return res_def

    def _show_entire_graph(graph_def, max_const_size=32):
        """Visualize TensorFlow graph."""
        if hasattr(graph_def, 'as_graph_def'):
            graph_def = graph_def.as_graph_def()
        strip_def = _strip_consts(graph_def, max_const_size=max_const_size)
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="{baseURL}/files/tf-graph/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()), baseURL=baseURL)

        iframe = """
            <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))

        return iframe

    # Visualizing the network graph. Be sure expand the "mixed" nodes to see their
    # internal structure. We are going to visualize "Conv2D" nodes.
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    tmp_def = graph_def
    # tmp_def = _rename_nodes(graph_def, lambda s: "/".join(s.split('.', 1)))
    return _show_entire_graph(tmp_def)

def show_graph(graph_def):
    iframe = graph_as_HTML(graph_def)
    display(HTML(iframe))

def writeAsHTML(graphdefFile, htmlFile):
    graph_def = tf.GraphDef()

    proto_b = open(graphdefFile, "rb").read()
    graph_def.ParseFromString(proto_b)

    html = graph_as_HTML(graph_def, baseURL='http://localhost:8889')
    open(htmlFile, 'w').write(html)

if __name__ == '__main__':
    import sys
    writeAsHTML(sys.argv[1], sys.argv[2])

