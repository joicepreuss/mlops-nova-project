# """
# This is a boilerplate pipeline
# generated using Kedro 0.18.8
# """

# from typing import Dict

# from kedro.pipeline import Pipeline, node

# from mlops_house_pricing.pipelines import data_modeling as dm
# from mlops_house_pricing.pipelines import data_processing as dp
# from mlops_house_pricing.data_quality.nodes import check_ranges


# ###########################################################################
# # Here you can find an example pipeline, made of two modular pipelines.
# #
# # Delete this when you start working on your own Kedro project as
# # well as pipelines/data_science AND pipelines/data_engineering
# # -------------------------------------------------------------------------


# def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
#     """Create the project's pipeline.

#     Args:
#         kwargs: Ignore any additional arguments added in the future.

#     Returns:
#         A mapping from a pipeline name to a ``Pipeline`` object.

#     """

#     data_processing_pipeline = dp.create_pipeline()
#     data_modeling_pipeline = dm.create_pipeline()
    
#      #### IMPLEMENT THIS ####
#     return {
#         "__default__": 

#         Pipeline(
#             [
#                 node(
#                     check_critical_nulls,
#                     inputs="",
#                     output=""
#                 )
#             ]
#         ) +

#         data_processing_pipeline +

#         Pipeline(
#             [
#                 node(
#                     check_ranges,
#                     inputs="",
#                     outputs="",
#                 )
#             ]
#         ) +

#         data_modeling_pipeline
#     }
#     #### IMPLEMENT THIS ####