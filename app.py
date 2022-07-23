import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn import tree 
from sklearn.preprocessing import LabelEncoder
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd


df = pd.read_csv('Data1.csv')
lb = LabelEncoder()
df['Outlook'] = lb.fit_transform(df['Outlook'])
df['Windy'] = lb.fit_transform(df['Windy'])
#df['Play_'] = lb.fit_transform(df['Play'])
X = df.iloc[:,1:5]
Y = df.iloc[:,5]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train) 
app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout  = dbc.Container([
     dbc.Row(
        dbc.Col(html.H1("Golf Dataset Classification",
                        className='text-center text-dark mb-4'),
                width=12)
    ),
       dbc.Row([
       
        dbc.Col([
           
            dbc.Card([
        
        dbc.CardBody(
            [
                 html.H2("9/5", className="card-title"),
                 html.P(
                "Train/ Test Split",
                className="card-text")
                
            ]
        ),
    ],
    style={"size": 3, "order": 1, "offset": 2}, className="card text-white card-body bg-dark me-1 border m-4",
)
            
        ]),
        dbc.Col([
            dbc.Card([
        
        dbc.CardBody(
            [
                html.H2("60%", className="card-title"),
                 html.P(
                "Model Test Accuracy",
                className="card-text")
                
            ]
        ),
    ],
    style={"size": 3, "order": 1, "offset": 2}, className="card text-white card-body bg-dark  me-1 border m-4 ",
)])
  
    ]),

    dbc.Row(
            [
        dbc.Col(
            [
                dbc.Label("Temperature"),
                dbc.Input(id='input-on-submit2', type='text'),   
            ],
            width=6,
        ),
        
        dbc.Col(
            [
                dbc.Label("Humidity"),
                dbc.Input(id='input-on-submit3', type='text'),
            ],
            width=6,
        ),
    ],
    className="g-3",
    
),
    
    
    dbc.Row([
        dbc.Col([
            dbc.Label("Outlook "),
        dcc.Dropdown(
                        options=[
                            {'label': 'overcast', 'value': '0'},
                            {'label': 'rainy', 'value': '1'},
                            {'label': 'sunny', 'value': '2'}
                        ],
                        value='0',
                        id='input-on-submit1'
        
             
        ),
        ]),
        dbc.Col([
            dbc.Label("Windy"),
        dcc.Dropdown(
                        options=[
                            {'label': 'True', 'value': '1'},
                            {'label': 'False', 'value': '0'}
                        ],
                        value='0',
                        id='input-on-submit4'
        ),
        
    
        ]),
    
    dbc.Row([
        dbc.Col([
              #dbc.Button("Submit",id='submit-val', className="me-1 border m-4"),
              
               dbc.Card(
    [
        dbc.CardImg(src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS6mv24FQR_mimXrdf6yN1ZQdbD8PKIcnnX1w&usqp=CAU", top=True),
        dbc.CardBody(
            [
                
                html.Div(id='container-button-basic',
              children='Enter a value and press submit'),
                dbc.Button("Submit",id='submit-val', color="dark",n_clicks=0),
            ]
        ),
    ],
    style={"width": "24"},
)
        ],className="d-grid gap-2 col-4 mx-auto  border m-4")
        
    ]),
 
        
    
    ]),


])



@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit1', 'value'),
    State('input-on-submit2', 'value'),
    State('input-on-submit3', 'value'),
    State('input-on-submit4', 'value'))


def update_output(n_clicks, value1, value2, value3, value4):
    test1 = [value1, value2, value3, value4]
    
    
    return 'The prediction result is  {} '.format(
        list(model.predict( [test1] ))
    )



if __name__ == "__main__":
    app.run_server(host='127.0.0.1', port='7080')   