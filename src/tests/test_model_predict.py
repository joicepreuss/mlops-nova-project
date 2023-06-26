import pickle
import numpy as np
import pandas as pd

from src.mlops_house_pricing.pipelines.model_predict.nodes import model_predict

def test_model_predict():
    """
    Test the model_predict function.
    """

    # Create dummy data for testing
    data = [
        ['60', 'RL', '65', '8450', 'Pave', 'NA', 'Reg', 'Lvl', 'AllPub', 'Inside', 'Gtl', 'CollgCr', 'Norm', 'Norm', '1Fam', '2Story', '7', '5', '2003', '2003', 'Gable', 'CompShg', 'VinylSd', 'VinylSd', 'BrkFace', '196', 'Gd', 'TA', 'PConc', 'Gd', 'TA', 'No', 'GLQ', '706', 'Unf', '0', '150', '856', 'GasA', 'Ex', 'Y', 'SBrkr', '856', '854', '0', '1710', '1', '0', '2', '1', '3', '1', 'Gd', '8', 'Typ', '0', 'NA', 'Attchd', '2003', 'RFn', '2', '548', 'TA', 'TA', 'Y', '0', '61', '0', '0', '0', '0', 'NA', 'NA', 'NA', '0', '2', '2008', 'WD', 'Normal'],
        ['20', 'RL', '80', '9600', 'Pave', 'NA', 'Reg', 'Lvl', 'AllPub', 'FR2', 'Gtl', 'Veenker', 'Feedr', 'Norm', '1Fam', '1Story', '6', '8', '1976', '1976', 'Gable', 'CompShg', 'MetalSd', 'MetalSd', 'None', '0', 'TA', 'TA', 'CBlock', 'Gd', 'TA', 'Gd', 'ALQ', '978', 'Unf', '0', '284', '1262', 'GasA', 'Ex', 'Y', 'SBrkr', '1262', '0', '0', '1262', '0', '1', '2', '0', '3', '1', 'TA', '6', 'Typ', '1', 'TA', 'Attchd', '1976', 'RFn', '2', '460', 'TA', 'TA', 'Y', '298', '0', '0', '0', '0', '0', 'NA', 'NA', 'NA', '0', '5', '2007', 'WD', 'Normal'],
        ['60', 'RL', '68', '11250', 'Pave', 'NA', 'IR1', 'Lvl', 'AllPub', 'Inside', 'Gtl', 'CollgCr', 'Norm', 'Norm', '1Fam', '2Story', '7', '5', '2001', '2002', 'Gable', 'CompShg', 'VinylSd', 'VinylSd', 'BrkFace', '162', 'Gd', 'TA', 'PConc', 'Gd', 'TA', 'Mn', 'GLQ', '486', 'Unf', '0', '434', '920', 'GasA', 'Ex', 'Y', 'SBrkr', '920', '866', '0', '1786', '1', '0', '2', '1', '3', '1', 'Gd', '6', 'Typ', '1', 'TA', 'Attchd', '2001', 'RFn', '2', '608', 'TA', 'TA', 'Y', '0', '42', '0', '0', '0', '0', 'NA', 'NA', 'NA', '0', '9', '2008', 'WD', 'Normal'],
        ['70', 'RL', '60', '9550', 'Pave', 'NA', 'IR1', 'Lvl', 'AllPub', 'Corner', 'Gtl', 'Crawfor', 'Norm', 'Norm', '1Fam', '2Story', '7', '5', '1915', '1970', 'Gable', 'CompShg', 'Wd Sdng', 'Wd Shng', 'None', '0', 'TA', 'TA', 'BrkTil', 'TA', 'Gd', 'No', 'ALQ', '216', 'Unf', '0', '540', '756', 'GasA', 'Gd', 'Y', 'SBrkr', '961', '756', '0', '1717', '1', '0', '1', '0', '3', '1', 'Gd', '7', 'Typ', '1', 'Gd', 'Detchd', '1998', 'Unf', '3', '642', 'TA', 'TA', 'Y', '0', '35', '272', '0', '0', '0', 'NA', 'NA', 'NA', '0', '2', '2006', 'WD', 'Abnorml'],
        ['60', 'RL', '84', '14260', 'Pave', 'NA', 'IR1', 'Lvl', 'AllPub', 'FR2', 'Gtl', 'NoRidge', 'Norm', 'Norm', '1Fam', '2Story', '8', '5', '2000', '2000', 'Gable', 'CompShg', 'VinylSd', 'VinylSd', 'BrkFace', '350', 'Gd', 'TA', 'PConc', 'Gd', 'TA', 'Av', 'GLQ', '655', 'Unf', '0', '490', '1145', 'GasA', 'Ex', 'Y', 'SBrkr', '1145', '1053', '0', '2198', '1', '0', '2', '1', '4', '1', 'Gd', '9', 'Typ', '1', 'TA', 'Attchd', '2000', 'RFn', '3', '836', 'TA', 'TA', 'Y', '192', '84', '0', '0', '0', '0', 'NA', 'NA', 'NA', '0', '12', '2008', 'WD', 'Normal']
    ]

    columns = [
        'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',
        'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
        'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
        'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
        'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
        'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
    ]

    X = pd.DataFrame(data, columns=columns)#[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MoSold', 'YrSold']]

    # Read a fake pickles
    with open("./src/tests/sample/cleaning_preprocessor.pkl", 'rb') as file:
        cleaning_preprocessor = pickle.load(file)

    with open("./src/tests/sample/feat_eng_preprocessor.pkl", 'rb') as file:
        feat_eng_preprocessor = pickle.load(file)
    
    with open("./src/tests/sample/trained_model.pkl", 'rb') as file:
        model = pickle.load(file)

    # Call the function to make predictions
    predictions, describe_servings = model_predict(X, cleaning_preprocessor, feat_eng_preprocessor, model)
    
    # Assert return is not None
    assert predictions is not None
    assert describe_servings is not None
    
    # Assert return type
    assert isinstance(predictions, pd.DataFrame)
    assert isinstance(describe_servings, dict)
    
    # Assert return size
    assert len(predictions) == len(X)
    
    # Assert return columns
    assert "y_pred" in predictions.columns

test_model_predict()