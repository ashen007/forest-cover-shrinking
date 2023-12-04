import graphviz

from torchview import draw_graph
from forest_cover_change_detection.models.fcef.modules import *
from forest_cover_change_detection.models.fcfe_with_att.modules import *

from forest_cover_change_detection.models.fcfe_with_att.fcfe_att import FCFEWithAttention


def visual_graph(model, input_size, name, path, expand_nested, direction):
    model_graph = draw_graph(model,
                             graph_dir=direction,
                             graph_name=name,
                             save_graph=True,
                             directory=path,
                             input_size=input_size,
                             expand_nested=expand_nested)

    return model_graph.visual_graph


def render_svg(name, path, src_file):
    g = graphviz.Source.from_file(f'{path}/{src_file}')
    g.render(f'{path}/{name}', format='svg')


if __name__ == '__main__':
    t = (16, 6, 256, 256)
    t_ = (16, 16, 8, 8)

    model = FCFEWithAttention(6, 2)

    g = visual_graph(model, t,
                     'fcef_att',
                     './runs/att',
                     True,
                     'TB')
    # render_svg('fcef', './runs/fcef_res', 'fcef_res.gv')
