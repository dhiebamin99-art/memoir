# LSTM — sur la série agrégée
    # Conseil : filtrer sur un article fréquent pour un LSTM plus précis
    # ex : train_lstm(df_feat, article="ART001", n_steps=30, epochs=50)
print("\n--- LSTM ---")
lstm_res = train_lstm(df_feat, article=None, n_steps=30, epochs=50, n_test=30)