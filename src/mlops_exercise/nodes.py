
import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def clean_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Does dome data cleaning.
    Args:
        data: Data containing features and target.
    Returns:
        data: Cleaned data
    """
    #remove some outliers
    df_transformed = data.copy()

    # describe_to_dict = df_transformed.describe().to_dict()

    # for cols in ["age"]:
    #     Q1 = df_transformed[cols].quantile(0.25)
    #     Q3 = df_transformed[cols].quantile(0.75)
    #     IQR = Q3 - Q1     

    # filter = (df_transformed[cols] >= Q1 - 1.5 * IQR) & (df_transformed[cols] <= Q3 + 1.5 *IQR)
    # df_transformed = df_transformed.loc[filter]
    
    # #we can do some basic cleaning by impuation of all null values
    # df_transformed.fillna(-9999,inplace=True)

    # describe_to_dict_verified = df_transformed.describe().to_dict()
    df_transformed = df_transformed.head(5)
    
    logger.info(f"The final dataframe has {len(df_transformed.columns)} columns.")

    return df_transformed#, describe_to_dict, describe_to_dict_verified 