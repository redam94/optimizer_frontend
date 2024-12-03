import streamlit as st
from streamlit_elements import elements, mui, html
from budget_optimizer.utils.model_classes import BaseBudgetModel
from budget_optimizer.optimizer import  Optimizer
from pydantic import BaseModel
import scipy.optimize as opt
from pathlib import Path
import numpy as np
import yaml

class RevenueModel(BaseBudgetModel):
    def __init__(self, model_name: str, model_kpi: str, model_path: str):
        super().__init__(model_name, model_kpi, model_path)



MODEL_NAME = "Revenue Model"
MODEL_KPI = "Revenue"
MODEL_PATH = Path(__file__).parent /"../models"
model = RevenueModel(MODEL_NAME, MODEL_KPI, MODEL_PATH)

with open(Path(__file__).parent/"../models/optimizer_config.yaml", "r") as file:
    config = yaml.safe_load(file)

initial_budget = config["initial_budget"]
init_budget = np.zeros(len(initial_budget))
bounds = [(-.2, .2) for _ in range(len(initial_budget))]
constraints = opt.LinearConstraint([list(initial_budget.values())], [0], [0])
optimizer = Optimizer(model, MODEL_PATH)


if "optimized_budget" not in st.session_state:
    st.session_state.optimized_budget = {channel: value for channel, value in initial_budget.items()}
    
def get_channels():
    return list(initial_budget.keys())

def get_initial_budgets():
    return list(initial_budget.values())

def run_optimizer(initial_settings):
    init_budget = np.array([(setting['starting_budget'] - initial_budget[channel])/initial_budget[channel] for channel, setting in initial_settings.items()])
    bounds = [(setting['lower_bound'], setting['upper_bound']) for channel, setting in initial_settings.items()]
    constraints = opt.LinearConstraint([list(initial_budget.values())], [0], [0])
    optimizer.optimize(init_budget, bounds, constraints)
    st.session_state.optimized_budget = optimizer.optimal_budget
    
    
with st.sidebar:
    with st.form(key='my_form', clear_on_submit=False):
        st.write('Please enter your budget for each channel')
        temp_dict = {}
        for channel, budget in zip(get_channels(), get_initial_budgets()):
            temp_dict[channel] = {'name': channel}
            with st.container():
                st.write(channel)
                cols = st.columns([1, 1])
                temp_dict[channel]['starting_budget']= cols[0].number_input('Initial Budget ($K)', value=budget, min_value=0.0, step=10.0, key=f"{channel}_initial_budget")
                temp_dict[channel]['lower_bound'], temp_dict[channel]['upper_bound'] = cols[1].slider('Budget % Range', value=(-.2, .2), min_value=-1.0, max_value=1.0, key=f"{channel}_lower_bound")
            
        submit_button = st.form_submit_button(label='Submit', on_click= lambda: run_optimizer(temp_dict))

with elements("dashboard"):

    # You can create a draggable and resizable dashboard using
    # any element available in Streamlit Elements.

    from streamlit_elements import dashboard

    # First, build a default layout for every element you want to include in your dashboard

    layout = [
        # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
        dashboard.Item("first_item", 0, 0, 2, 2),
        #dashboard.Item("second_item", 2, 0, 2, 2,),
        #dashboard.Item("third_item", 0, 2, 1, 1),
    ]

  

    # If you want to retrieve updated layout values as the user move or resize dashboard items,
    # you can pass a callback to the onLayoutChange event parameter.

    def handle_layout_change(updated_layout):
        # You can save the layout in a file, or do anything you want with it.
        # You can pass it back to dashboard.Grid() if you want to restore a saved layout.
        print(updated_layout)

    with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
      from streamlit_elements import nivo
      Total_Budget = 100
      DATA = [{"channel": channel, 'initial budget': budget/sum(initial_budget.values())*100, 'optimal budget': st.session_state.optimized_budget[channel]/sum(initial_budget.values())*100} for channel, budget in initial_budget.items()]
    #   DATA = [
    #     { "channel": "OLV", "initial budget": 20, "optimized budget": 24},
    #     { "channel": "Social", "initial budget": 5, "optimized budget": 6},
    #     { "channel": "Display", "initial budget": 30, "optimized budget": 25},
    #     { "channel": "Search", "initial budget": 35, "optimized budget":  29},
    #     { "channel": "Audio", "initial budget": 10, "optimized budget": 16},
    #   ]

      with mui.Box(sx={"height": 500}, key="first_item"):
        nivo.Radar(
              data=DATA,
              keys=[ "initial budget", 'optimal budget'],
              indexBy="channel",
              maxValue=80,
              valueFormat=">-.0f",
              margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
              borderColor={ "from": "color" },
              gridLabelOffset=36,
              dotSize=10,
              dotColor={ "theme": "background" },
              dotBorderWidth=2,
              motionConfig="wobbly",
              legends=[
                  {
                      "anchor": "top-left",
                      "direction": "column",
                      "translateX": -50,
                      "translateY": -40,
                      "itemWidth": 80,
                      "itemHeight": 20,
                      "itemTextColor": "#999",
                      "symbolSize": 12,
                      "symbolShape": "circle",
                      "effects": [
                          {
                              "on": "hover",
                              "style": {
                                  "itemTextColor": "#000"
                              }
                          }
                      ]
                  }
              ],
              theme={
                  "background": "#FFFFFF",
                  "textColor": "#31333F",
                  "tooltip": {
                      "container": {
                          "background": "#FFFFFF",
                          "color": "#31333F",
                      }
                  }
              })
      