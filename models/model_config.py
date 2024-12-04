import xarray as xr
from pathlib import Path
import numpy as np
from budget_optimizer.utils.model_helpers import AbstractModel, BudgetType

INITIAL_BUDGET: BudgetType = dict(OLV=25000.0, Social=7000.0, Display=3000.0, Search=3500.0, Audio=1000.0)

class SimpleModel(AbstractModel):
  """
  Simple model that just adds the two variables a and b.
  This can be as complex as you want as long as it has a predict method
  that takes an xarray Dataset and returns an xarray DataArray and 
  a contributions method that takes an xarray Dataset and returns an xarray Dataset.
  
  Ideally, the model should also have data that defines the initial data that the
  model was trained on. You can wrap cutom models or functions in a class like this.
  """
  def __init__(self, data: xr.Dataset = None):
    self.data = data
  
  def _shape_transform(self, x: xr.DataArray, N: float, k: float) -> xr.DataArray:
    return x**k/(x**k + N**k)
    
  def predict(self, x: xr.Dataset) -> xr.DataArray:
    x = x.copy()
    olv_contribution = self._shape_transform(x["OLV"], 1.2, 2)
    social_contribution = self._shape_transform(x["Social"], 1.5, 3)
    display_contribution = self._shape_transform(x["Display"], 1.8, 4)
    search_contribution = self._shape_transform(x["Search"], 2.1, 5)
    audio_contribution = self._shape_transform(x["Audio"], 2.4, 6)
    t = x["trend"]
    
    x["prediction"] = np.exp( 
        .5*np.log(t+1) 
        + .2*olv_contribution
        + .1*social_contribution
        + .05*display_contribution
        + .15*search_contribution
        + .1*audio_contribution
        + np.log(1000))
    return x["prediction"]
  
  def contributions(self, x: xr.Dataset) -> xr.Dataset:
    x = x.copy()
    olv_contribution = self._shape_transform(x["OLV"], 1.2, 2)
    social_contribution = self._shape_transform(x["Social"], 1.5, 3)
    display_contribution = self._shape_transform(x["Display"], 1.8, 4)
    search_contribution = self._shape_transform(x["Search"], 2.1, 5)
    audio_contribution = self._shape_transform(x["Audio"], 2.4, 6)
    t = x["trend"]
    prediction = np.exp( 
        .5*np.log(t+1) 
        + .2*olv_contribution
        + .1*social_contribution
        + .05*display_contribution
        + .15*search_contribution
        + .1*audio_contribution
        + np.log(1000))
    
    x["OLV"] = prediction - prediction/np.exp(.2*olv_contribution)
    x["Social"] = prediction - prediction/np.exp(.1*social_contribution)
    x["Display"] = prediction - prediction/np.exp(.05*display_contribution)
    x["Search"] = prediction - prediction/np.exp(.15*search_contribution)
    x["Audio"] = prediction - prediction/np.exp(.1*audio_contribution)
    return x

def budget_to_data(budget: BudgetType, model: AbstractModel) -> xr.Dataset:
    data = model.data.copy()
    for key, value in budget.items():
        data[key] = value/INITIAL_BUDGET[key]*data[key]
    return data
  
def model_loader(path: Path) -> AbstractModel:
    rng = np.random.default_rng(42)
    trend = xr.DataArray(np.linspace(0, 3, 156), dims='time', coords={"time": np.arange(1, 157)})
    
    data_olv = xr.DataArray(np.exp(rng.normal(-2, 4, size=156)), dims='time', coords={"time": np.arange(1, 157)})
    data_social = xr.DataArray(np.exp(rng.normal(-2, 2.4, size=156)), dims='time', coords={"time": np.arange(1, 157)})
    data_display = xr.DataArray(np.exp(rng.normal(-2, 2, size=156)), dims='time', coords={"time": np.arange(1, 157)})
    data_search = xr.DataArray(np.exp(rng.normal(-2, 2.2, size=156)), dims='time', coords={"time": np.arange(1, 157)})
    data_audio = xr.DataArray(np.exp(rng.normal(-2, 1.8, size=156)), dims='time', coords={"time": np.arange(1, 157)})
    return SimpleModel(
      data = xr.Dataset({
        "OLV": data_olv, 
        "Social": data_social, 
        "Display": data_display, 
        "Search": data_search, 
        "Audio": data_audio, 
        "trend": trend}))