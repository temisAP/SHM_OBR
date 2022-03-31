# %%

""" Load model """

from keras.models import load_model

model_T = load_model('../2_models/model_T.h5')
model_P = load_model('../2_models/model_P.h5')

# %%

""" Predict model """

Y_pred_T = model_T.predict(X_test)
Y_pred_P = model_P.predict(X_test)

# %%

""" Evaluate model """

score_T = model_T.evaluate(X_test, Y_test[:,0], batch_size=32)
score_P = model_P.evaluate(X_test, Y_test[:,1], batch_size=32)

# %%

""" Predict model """

Y_pred_T = model_T.predict(X_test)
Y_pred_P = model_P.predict(X_test)

# %%

""" Evaluate model """

score_T = model_T.evaluate(X_test, Y_test[:,0], batch_size=32)
score_P = model_P.evaluate(X_test, Y_test[:,1], batch_size=32)

# %%

""" Predict model """

Y_pred_T = model_T.predict(X_test)
Y_pred_P = model_P.predict(X_test)

# %%
