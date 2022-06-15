from pyecharts.charts import Funnel
from pyecharts.charts import Grid
import pyecharts.options as opts
import pandas as pd
import random

dataset = pd.read_csv('/Users/xuhongtao/PycharmProjects/Resource/12-16.csv')

random_date = random.choices(dataset['time'])
random_stock = dataset[(dataset['time'] == random_date[0])][['BK300_name', 'closing']]
random_stock = random_stock.sample(5)

random_stock.sort_values(by='closing', inplace=True, ascending=False)

x_data = random_stock['BK300_name'].to_list()
y_data = random_stock['closing'].to_list()

data = [[x_data[i], y_data[i]] for i in range(len(x_data))]

e_x_data = random_stock['BK300_name'].to_list()
e_y_data = [100, 80, 60, 40, 20]

ex_data = [[e_x_data[i], e_y_data[i]] for i in range(len(e_x_data))]

funnel = (
    Funnel(init_opts=opts.InitOpts(width="1600px", height="800px"))
    .add(
        series_name="expected",
        data_pair=ex_data,
        gap=2,
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}%"),
        label_opts=opts.LabelOpts(is_show=True, position="inside"),
        itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1, opacity=0.5),
    )
)

funnel1 = (
    Funnel(init_opts=opts.InitOpts(width="1600px", height="800px"))
    .add(
        series_name="AAA",
        data_pair=data,
        gap=2,
        tooltip_opts=opts.TooltipOpts(trigger="item"),
        label_opts=opts.LabelOpts(is_show=True, position="top"),
        itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1, opacity=0.7),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="%s收市价格漏斗图" % random_date[0], subtitle="芜湖~"))

)

grid = (
    Grid()
    .add(funnel1, opts.GridOpts(pos_left="50%", pos_right="50%"), is_control_axis_index=True)
    .add(funnel, opts.GridOpts(pos_left="50%", pos_right="50%"), is_control_axis_index=True)

)

grid.load_javascript()
grid.render('111.html')