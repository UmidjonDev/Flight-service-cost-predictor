#Importing essential libraries
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import dill

#Dropping unnecessary columns from the dataset
def drop_unimportant(df : pd.DataFrame) -> pd.DataFrame : 
    return df.drop(columns = ['id', 'airline'])

#Basic feature changer
def change_features(df : pd.DataFrame) -> pd.DataFrame :
    df['flight'] = df['flight'].apply(lambda x : x[0 : 2])
    df['stops'] = df['stops'].apply(lambda x : 0 if (x == 'zero') else 1 if (x == 'one') else 2)
    return df

def pipeline() -> None :
    print('Flight cost predictor pipeline !')

    #Data loading
    df = pd.read_csv(filepath_or_buffer = './data/train_data.csv', sep = ",")
    df.head()

    # Preprocess the entire DataFrame first
    X = df.drop(columns = 'price')
    y = df[['price']]
    scale_up = StandardScaler()
    y_scaled = scale_up.fit_transform(X = y)

    #Feature engineering
    ohe_cols = ['source_city', 'departure_time', 'arrival_time', 'destination_city', 'class', 'flight']
    std_scaler = ['duration', 'days_left']
    stop_cols = ['stops']

    first_feature_engineering = Pipeline(steps = [
        ('drop_cols', FunctionTransformer(drop_unimportant)),
        ('change_features', FunctionTransformer(change_features))
    ])
    numerical_transformer = Pipeline(steps = [
        ('scaler', StandardScaler())
    ])
    ohe_transformation = Pipeline(steps = [
        ('ohe', OneHotEncoder(handle_unknown = 'ignore'))
    ])
    remaining_transformation = Pipeline(steps = [
            ('remaining_features', FunctionTransformer(lambda x : x))
    ])
    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, std_scaler),
        ('ohe_transformation', ohe_transformation, ohe_cols),
        ('remaining_features', remaining_transformation, stop_cols)
    ])
    preprocessor = Pipeline(steps = [
        ('feature_change', first_feature_engineering),
        ('column_transformer', column_transformer)
    ])

    model = RandomForestRegressor(random_state = 1, n_estimators = 100, min_samples_split = 20, min_samples_leaf = 8, bootstrap = True)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    #Fitting perfect pipeline for whole dataset
    pipe.fit(X = X, y = y_scaled)

    pred = pipe.predict(X = X)

    print(f"Mean absolute error of Random Forest Regressor algorithm in train dataset is {mean_absolute_error(y_true = y_scaled, y_pred = pred)}")
    print(f"Mean squared error of Random Forest Regressor algorithm in train dataset is {mean_squared_error(y_true = y_scaled, y_pred = pred)}")
    print(f"R2 score of Random Forest Regressor algorithm in train dataset is {r2_score(y_true = y_scaled, y_pred = pred)}")
    
    model_filename = f'./models/flight_cost.pkl'
    dill.dump({'model' : pipe,
        'metadata' :{
            'name' : 'Flight cost predictor',
            'author' : 'Umidjon Sattorov',
            'version' : 1,
            'date' : datetime.now(),
            'type' : type(pipe.named_steps['regressor']).__name__,
            'r2_score' : r2_score(y_true = y_scaled, y_pred = pred)
        }
    }, open('./models/flight_cost.pkl', 'wb'))

    print(f'Model is saved as {model_filename} in models directory')

if __name__ == '__main__':
    pipeline()
