# Text-FCG
--------------------------------------------------
params: [start=0, batch_size=2560, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18146, 300, padding_idx=18145)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.5, inplace=False)
  )
  (gcn): GraphLayer(
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (dropout): Dropout(p=0.5, inplace=False)
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
C:\Anaconda3\lib\site-packages\torch\nn\functional.py:1628: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.
  warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
epoch=001, cost=15.57, train:[0.7092, 50.37%], valid:[0.6355, 50.14%], test:[0.7049, 47.68%], best_acc=47.68%
epoch=002, cost=14.57, train:[0.6860, 53.81%], valid:[0.6750, 56.44%], test:[0.6693, 58.23%], best_acc=58.23%
epoch=003, cost=14.69, train:[0.6625, 60.68%], valid:[0.6266, 63.90%], test:[0.6345, 64.42%], best_acc=64.42%
epoch=004, cost=14.72, train:[0.6049, 65.77%], valid:[0.5973, 69.13%], test:[0.5696, 72.43%], best_acc=72.43%
epoch=005, cost=14.65, train:[0.5759, 70.06%], valid:[0.5683, 69.92%], test:[0.5455, 71.87%], best_acc=72.43%
epoch=006, cost=14.73, train:[0.5409, 71.92%], valid:[0.5484, 70.71%], test:[0.5537, 70.18%], best_acc=72.43%
epoch=007, cost=14.57, train:[0.5336, 73.03%], valid:[0.5232, 72.79%], test:[0.5244, 74.40%], best_acc=74.40%
epoch=008, cost=14.64, train:[0.5014, 76.00%], valid:[0.5017, 74.51%], test:[0.5069, 76.51%], best_acc=76.51%
epoch=009, cost=14.64, train:[0.4817, 76.50%], valid:[0.4980, 75.38%], test:[0.4723, 76.79%], best_acc=76.79%
epoch=010, cost=14.66, train:[0.4682, 77.16%], valid:[0.4776, 76.06%], test:[0.4650, 77.78%], best_acc=77.78%
epoch=011, cost=14.81, train:[0.4567, 78.33%], valid:[0.4743, 76.36%], test:[0.4792, 77.50%], best_acc=77.78%
epoch=012, cost=14.73, train:[0.4391, 79.30%], valid:[0.4668, 76.84%], test:[0.4572, 79.04%], best_acc=79.04%
epoch=013, cost=14.69, train:[0.4299, 79.69%], valid:[0.4665, 77.07%], test:[0.4564, 79.75%], best_acc=79.75%
epoch=014, cost=15.13, train:[0.4163, 80.55%], valid:[0.4822, 76.51%], test:[0.4694, 78.76%], best_acc=79.75%
epoch=015, cost=14.98, train:[0.4073, 81.21%], valid:[0.4757, 77.38%], test:[0.4501, 80.17%], best_acc=80.17%
epoch=016, cost=14.82, train:[0.3976, 81.73%], valid:[0.4720, 78.28%], test:[0.4451, 80.17%], best_acc=80.17%
epoch=017, cost=14.89, train:[0.3825, 82.98%], valid:[0.4637, 77.88%], test:[0.4552, 79.04%], best_acc=80.17%
epoch=018, cost=14.96, train:[0.3652, 83.49%], valid:[0.4635, 78.50%], test:[0.4579, 80.87%], best_acc=80.87%
epoch=019, cost=15.19, train:[0.3580, 84.52%], valid:[0.4721, 78.36%], test:[0.4581, 79.75%], best_acc=80.87%
epoch=020, cost=15.19, train:[0.3454, 84.41%], valid:[0.4658, 79.01%], test:[0.4351, 80.17%], best_acc=80.87%
epoch=021, cost=14.84, train:[0.3342, 85.90%], valid:[0.4760, 78.87%], test:[0.4447, 79.75%], best_acc=80.87%
epoch=022, cost=14.77, train:[0.3105, 86.60%], valid:[0.5278, 78.39%], test:[0.4675, 80.17%], best_acc=80.87%
epoch=023, cost=14.75, train:[0.3022, 86.54%], valid:[0.4719, 78.76%], test:[0.4620, 80.59%], best_acc=80.87%
epoch=024, cost=14.81, train:[0.2816, 88.18%], valid:[0.5098, 78.78%], test:[0.5116, 78.76%], best_acc=80.87%
epoch=025, cost=14.63, train:[0.2841, 88.07%], valid:[0.5622, 75.91%], test:[0.5552, 77.22%], best_acc=80.87%
epoch=026, cost=14.92, train:[0.2998, 87.04%], valid:[0.5274, 77.55%], test:[0.5052, 77.92%], best_acc=80.87%
epoch=027, cost=15.38, train:[0.2740, 88.56%], valid:[0.6069, 77.91%], test:[0.5968, 78.90%], best_acc=80.87%
epoch=028, cost=14.79, train:[0.2612, 88.56%], valid:[0.5840, 78.84%], test:[0.5203, 79.32%], best_acc=80.87%
epoch=029, cost=15.00, train:[0.2300, 90.07%], valid:[0.5113, 78.45%], test:[0.4803, 79.61%], best_acc=80.87%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.8404    0.7033    0.7658       337
           1     0.7669    0.8797    0.8194       374
    accuracy                         0.7961       711
   macro avg     0.8037    0.7915    0.7926       711
weighted avg     0.8017    0.7961    0.7940       711
Macro average Test Precision, Recall and F1-Score...
(0.8036626494073302, 0.7914716196702581, 0.7925891799129676, None)
Micro average Test Precision, Recall and F1-Score...
(0.7960618846694796, 0.7960618846694796, 0.7960618846694796, None)



--------------------------------------------------
params: [start=0, batch_size=2560, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18146, 300, padding_idx=18145)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.5, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=1)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=10.01, train:[0.7574, 50.68%], valid:[0.7742, 50.06%], test:[0.7144, 48.24%], best_acc=48.24%
epoch=002, cost=8.56, train:[0.7009, 51.35%], valid:[0.6934, 50.11%], test:[0.6928, 51.76%], best_acc=51.76%
epoch=003, cost=8.55, train:[0.6902, 51.91%], valid:[0.6657, 67.64%], test:[0.6653, 68.07%], best_acc=68.07%
epoch=004, cost=8.59, train:[0.6612, 61.03%], valid:[0.6345, 67.78%], test:[0.6343, 68.35%], best_acc=68.35%
epoch=005, cost=8.55, train:[0.6170, 66.78%], valid:[0.5914, 69.19%], test:[0.5589, 72.29%], best_acc=72.29%
epoch=006, cost=8.63, train:[0.5748, 70.24%], valid:[0.5508, 71.30%], test:[0.5229, 73.56%], best_acc=73.56%
epoch=007, cost=8.56, train:[0.5378, 72.55%], valid:[0.5319, 72.68%], test:[0.5103, 75.11%], best_acc=75.11%
epoch=008, cost=8.60, train:[0.5189, 73.69%], valid:[0.5033, 73.83%], test:[0.5117, 76.37%], best_acc=76.37%
epoch=009, cost=8.66, train:[0.4931, 75.41%], valid:[0.4989, 74.37%], test:[0.4815, 77.64%], best_acc=77.64%
epoch=010, cost=8.59, train:[0.4759, 76.88%], valid:[0.4939, 75.07%], test:[0.4799, 76.79%], best_acc=77.64%
epoch=011, cost=8.58, train:[0.4599, 78.04%], valid:[0.4685, 76.42%], test:[0.4765, 77.92%], best_acc=77.92%
epoch=012, cost=8.65, train:[0.4431, 78.96%], valid:[0.4862, 77.18%], test:[0.4708, 78.62%], best_acc=78.62%
epoch=013, cost=8.68, train:[0.4318, 79.51%], valid:[0.4699, 77.27%], test:[0.4643, 78.20%], best_acc=78.62%
epoch=014, cost=8.57, train:[0.4210, 80.29%], valid:[0.4896, 76.93%], test:[0.4714, 78.48%], best_acc=78.62%
epoch=015, cost=8.57, train:[0.4107, 81.51%], valid:[0.4666, 77.15%], test:[0.4651, 78.90%], best_acc=78.90%
epoch=016, cost=8.51, train:[0.4068, 81.62%], valid:[0.4513, 77.27%], test:[0.4531, 78.76%], best_acc=78.90%
epoch=017, cost=8.57, train:[0.3885, 82.52%], valid:[0.4696, 77.86%], test:[0.4649, 78.06%], best_acc=78.90%
epoch=018, cost=8.61, train:[0.3850, 82.40%], valid:[0.4706, 78.17%], test:[0.4639, 79.32%], best_acc=79.32%
epoch=019, cost=8.57, train:[0.3581, 84.26%], valid:[0.4644, 77.80%], test:[0.4554, 78.90%], best_acc=79.32%
epoch=020, cost=8.63, train:[0.3463, 84.62%], valid:[0.4721, 78.31%], test:[0.4489, 78.90%], best_acc=79.32%
epoch=021, cost=8.58, train:[0.3298, 85.54%], valid:[0.4758, 78.36%], test:[0.4618, 79.18%], best_acc=79.32%
epoch=022, cost=8.55, train:[0.3247, 86.01%], valid:[0.4872, 78.36%], test:[0.4564, 79.89%], best_acc=79.89%
epoch=023, cost=8.56, train:[0.3079, 87.03%], valid:[0.5185, 78.73%], test:[0.4605, 81.29%], best_acc=81.29%
epoch=024, cost=8.55, train:[0.2908, 87.87%], valid:[0.5709, 78.02%], test:[0.5013, 79.04%], best_acc=81.29%
epoch=025, cost=8.58, train:[0.2764, 88.04%], valid:[0.5407, 78.33%], test:[0.4687, 80.59%], best_acc=81.29%
epoch=026, cost=8.58, train:[0.2735, 88.90%], valid:[0.5480, 77.15%], test:[0.4948, 78.76%], best_acc=81.29%
epoch=027, cost=8.61, train:[0.2741, 88.48%], valid:[0.5836, 75.83%], test:[0.5276, 77.50%], best_acc=81.29%
epoch=028, cost=8.61, train:[0.2802, 87.79%], valid:[0.5827, 77.83%], test:[0.5029, 79.18%], best_acc=81.29%
epoch=029, cost=8.61, train:[0.2367, 89.64%], valid:[0.6613, 78.05%], test:[0.5508, 79.89%], best_acc=81.29%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.7764    0.8587    0.8155       368
           1     0.8289    0.7347    0.7790       343
    accuracy                         0.7989       711
   macro avg     0.8027    0.7967    0.7972       711
weighted avg     0.8018    0.7989    0.7979       711
Macro average Test Precision, Recall and F1-Score...
(0.8026800724169145, 0.7966947648624667, 0.7972318891160194, None)
Micro average Test Precision, Recall and F1-Score...
(0.7988748241912799, 0.7988748241912799, 0.7988748241912799, None)


--------------------------------------------------
params: [start=0, batch_size=1024, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18384, 300, padding_idx=18383)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.5, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=1)
    (encode): Sequential(
      (0): Dropout(p=0.4, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=16.88, train:[0.6995, 54.42%], valid:[0.6564, 51.44%], test:[0.6649, 54.71%], best_acc=54.71%
epoch=002, cost=15.29, train:[0.6310, 61.95%], valid:[0.6237, 68.54%], test:[0.6240, 67.79%], best_acc=67.79%
epoch=003, cost=15.48, train:[0.5539, 71.86%], valid:[0.5204, 73.64%], test:[0.5028, 74.40%], best_acc=74.40%
epoch=004, cost=16.15, train:[0.4968, 75.49%], valid:[0.4739, 75.46%], test:[0.4559, 78.20%], best_acc=78.20%
epoch=005, cost=15.80, train:[0.4701, 77.86%], valid:[0.4685, 76.76%], test:[0.4597, 76.79%], best_acc=78.20%
epoch=006, cost=15.57, train:[0.4468, 78.93%], valid:[0.4699, 76.90%], test:[0.4295, 79.32%], best_acc=79.32%
epoch=007, cost=15.55, train:[0.4087, 81.02%], valid:[0.4487, 77.52%], test:[0.4509, 79.61%], best_acc=79.61%
epoch=008, cost=15.54, train:[0.3960, 82.34%], valid:[0.4637, 77.32%], test:[0.4279, 79.47%], best_acc=79.61%
epoch=009, cost=15.49, train:[0.3805, 83.18%], valid:[0.4438, 78.14%], test:[0.4199, 80.73%], best_acc=80.73%
epoch=010, cost=15.55, train:[0.3534, 84.24%], valid:[0.5543, 76.53%], test:[0.4909, 79.47%], best_acc=80.73%
epoch=011, cost=15.67, train:[0.3530, 84.29%], valid:[0.4429, 78.28%], test:[0.4168, 80.45%], best_acc=80.73%
epoch=012, cost=15.88, train:[0.3104, 87.12%], valid:[0.4689, 78.33%], test:[0.4327, 79.47%], best_acc=80.73%
epoch=013, cost=15.96, train:[0.2748, 88.65%], valid:[0.5185, 79.46%], test:[0.4991, 80.31%], best_acc=80.73%
epoch=014, cost=15.93, train:[0.2600, 89.06%], valid:[0.5637, 77.94%], test:[0.4956, 78.90%], best_acc=80.73%
epoch=015, cost=15.82, train:[0.2246, 90.53%], valid:[0.6318, 76.90%], test:[0.6356, 77.36%], best_acc=80.73%
epoch=016, cost=15.75, train:[0.2272, 90.40%], valid:[0.6520, 78.62%], test:[0.5995, 79.18%], best_acc=80.73%
epoch=017, cost=15.79, train:[0.1817, 92.48%], valid:[0.6424, 79.04%], test:[0.5633, 80.31%], best_acc=80.73%
epoch=018, cost=15.81, train:[0.1416, 94.33%], valid:[0.7759, 78.19%], test:[0.7372, 80.73%], best_acc=80.73%
epoch=019, cost=15.86, train:[0.1313, 94.90%], valid:[0.8964, 78.42%], test:[0.7842, 80.87%], best_acc=80.87%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.8235    0.8148    0.8191       378
           1     0.7923    0.8018    0.7970       333
    accuracy                         0.8087       711
   macro avg     0.8079    0.8083    0.8081       711
weighted avg     0.8089    0.8087    0.8088       711
Macro average Test Precision, Recall and F1-Score...
(0.8079071391167743, 0.8083083083083082, 0.8080819307716736, None)
Micro average Test Precision, Recall and F1-Score...
(0.8087201125175809, 0.8087201125175809, 0.8087201125175809, None)


--------------------------------------------------
params: [start=0, batch_size=1024, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18384, 300, padding_idx=18383)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.5, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=1)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=17.02, train:[0.7145, 52.79%], valid:[0.6696, 50.45%], test:[0.6942, 46.41%], best_acc=46.41%
epoch=002, cost=16.74, train:[0.6541, 60.79%], valid:[0.5804, 69.64%], test:[0.5209, 74.54%], best_acc=74.54%
epoch=003, cost=17.22, train:[0.5619, 70.72%], valid:[0.5280, 73.66%], test:[0.4811, 78.76%], best_acc=78.76%
epoch=004, cost=16.48, train:[0.5110, 75.39%], valid:[0.4875, 76.08%], test:[0.4127, 80.45%], best_acc=80.45%
epoch=005, cost=16.29, train:[0.4688, 77.05%], valid:[0.4698, 76.65%], test:[0.4003, 82.98%], best_acc=82.98%
epoch=006, cost=16.40, train:[0.4513, 78.38%], valid:[0.4613, 77.72%], test:[0.3834, 82.84%], best_acc=82.98%
epoch=007, cost=16.19, train:[0.4177, 80.26%], valid:[0.4855, 76.03%], test:[0.4123, 82.00%], best_acc=82.98%
epoch=008, cost=16.77, train:[0.4062, 81.44%], valid:[0.4991, 75.77%], test:[0.4103, 81.72%], best_acc=82.98%
epoch=009, cost=16.66, train:[0.3860, 83.18%], valid:[0.4546, 78.25%], test:[0.3737, 82.98%], best_acc=82.98%
epoch=010, cost=16.66, train:[0.3642, 83.74%], valid:[0.4488, 78.81%], test:[0.3512, 83.54%], best_acc=83.54%
epoch=011, cost=16.79, train:[0.3204, 85.60%], valid:[0.4611, 78.98%], test:[0.3537, 83.68%], best_acc=83.68%
epoch=012, cost=16.60, train:[0.3044, 86.35%], valid:[0.4875, 78.53%], test:[0.3731, 82.42%], best_acc=83.68%
epoch=013, cost=16.67, train:[0.2702, 88.03%], valid:[0.5522, 77.72%], test:[0.4448, 83.12%], best_acc=83.68%
epoch=014, cost=16.45, train:[0.2651, 88.40%], valid:[0.5488, 78.17%], test:[0.4256, 82.70%], best_acc=83.68%
epoch=015, cost=16.58, train:[0.2481, 89.04%], valid:[0.6064, 77.74%], test:[0.4457, 82.14%], best_acc=83.68%
epoch=016, cost=16.62, train:[0.2494, 89.56%], valid:[0.4852, 79.21%], test:[0.4032, 84.53%], best_acc=84.53%
epoch=017, cost=16.56, train:[0.1961, 91.57%], valid:[0.6427, 78.70%], test:[0.4578, 82.84%], best_acc=84.53%
epoch=018, cost=16.59, train:[0.1872, 92.17%], valid:[0.8156, 77.57%], test:[0.6609, 81.15%], best_acc=84.53%
epoch=019, cost=16.64, train:[0.1648, 93.04%], valid:[0.7007, 78.73%], test:[0.5204, 83.97%], best_acc=84.53%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.8497    0.7927    0.8202       328
           1     0.8321    0.8799    0.8553       383
    accuracy                         0.8397       711
   macro avg     0.8409    0.8363    0.8378       711
weighted avg     0.8402    0.8397    0.8391       711
Macro average Test Precision, Recall and F1-Score...
(0.840885984023239, 0.8362892440934853, 0.8377596118432642, None)
Micro average Test Precision, Recall and F1-Score...
(0.8396624472573839, 0.8396624472573839, 0.8396624472573839, None)


--------------------------------------------------
params: [start=0, batch_size=1024, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18384, 300, padding_idx=18383)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.5, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=4)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=18.29, train:[0.7085, 51.87%], valid:[0.6738, 55.77%], test:[0.6777, 55.70%], best_acc=55.70%
epoch=002, cost=16.93, train:[0.6508, 63.22%], valid:[0.5920, 67.61%], test:[0.5709, 66.81%], best_acc=66.81%
epoch=003, cost=17.13, train:[0.5610, 70.63%], valid:[0.5245, 72.23%], test:[0.5176, 71.87%], best_acc=71.87%
epoch=004, cost=16.92, train:[0.4992, 75.47%], valid:[0.4921, 75.13%], test:[0.4926, 77.22%], best_acc=77.22%
epoch=005, cost=16.85, train:[0.4657, 77.72%], valid:[0.4983, 75.72%], test:[0.4798, 76.93%], best_acc=77.22%
epoch=006, cost=16.26, train:[0.4621, 78.16%], valid:[0.4607, 76.45%], test:[0.4404, 78.34%], best_acc=78.34%
epoch=007, cost=16.03, train:[0.4251, 80.65%], valid:[0.4755, 76.45%], test:[0.4339, 77.92%], best_acc=78.34%
epoch=008, cost=15.96, train:[0.4015, 81.57%], valid:[0.4489, 77.86%], test:[0.4245, 80.45%], best_acc=80.45%
epoch=009, cost=16.92, train:[0.3736, 83.46%], valid:[0.4989, 76.62%], test:[0.4470, 79.47%], best_acc=80.45%
epoch=010, cost=16.36, train:[0.3672, 83.46%], valid:[0.4606, 78.14%], test:[0.4188, 80.73%], best_acc=80.73%
epoch=011, cost=15.95, train:[0.3431, 84.57%], valid:[0.4827, 78.39%], test:[0.4186, 82.14%], best_acc=82.14%
epoch=012, cost=16.36, train:[0.3221, 86.78%], valid:[0.4893, 78.47%], test:[0.4081, 82.98%], best_acc=82.98%
epoch=013, cost=17.40, train:[0.2815, 88.28%], valid:[0.5376, 78.08%], test:[0.4890, 80.03%], best_acc=82.98%
epoch=014, cost=17.12, train:[0.2809, 88.07%], valid:[0.5026, 78.67%], test:[0.4545, 82.84%], best_acc=82.98%
epoch=015, cost=17.16, train:[0.2381, 90.12%], valid:[0.6293, 77.52%], test:[0.5115, 80.73%], best_acc=82.98%
epoch=016, cost=16.96, train:[0.2113, 91.79%], valid:[0.5830, 78.08%], test:[0.5337, 80.17%], best_acc=82.98%
epoch=017, cost=17.00, train:[0.1916, 92.42%], valid:[0.8044, 78.56%], test:[0.6951, 81.01%], best_acc=82.98%
epoch=018, cost=17.11, train:[0.1627, 93.51%], valid:[0.7226, 77.88%], test:[0.5641, 81.43%], best_acc=82.98%
epoch=019, cost=17.45, train:[0.1290, 95.48%], valid:[0.9168, 76.34%], test:[0.6734, 81.72%], best_acc=82.98%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.7937    0.8563    0.8238       355
           1     0.8445    0.7781    0.8099       356
    accuracy                         0.8172       711
   macro avg     0.8191    0.8172    0.8169       711
weighted avg     0.8192    0.8172    0.8169       711
Macro average Test Precision, Recall and F1-Score...
(0.8191229382920461, 0.8172139579047317, 0.8168948794751105, None)
Micro average Test Precision, Recall and F1-Score...
(0.8171589310829818, 0.8171589310829818, 0.8171589310829818, None)

--------------------------------------------------
params: [start=0, batch_size=1024, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18384, 300, padding_idx=18383)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.5, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=3)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=18.76, train:[0.6893, 52.51%], valid:[0.6556, 68.09%], test:[0.6504, 68.92%], best_acc=68.92%
epoch=002, cost=17.63, train:[0.6099, 68.09%], valid:[0.5925, 68.94%], test:[0.5680, 71.17%], best_acc=71.17%
epoch=003, cost=17.35, train:[0.5354, 73.33%], valid:[0.5097, 73.30%], test:[0.5053, 73.98%], best_acc=73.98%
epoch=004, cost=17.29, train:[0.4940, 75.79%], valid:[0.4874, 75.32%], test:[0.4769, 76.51%], best_acc=76.51%
epoch=005, cost=17.70, train:[0.4603, 78.10%], valid:[0.4826, 75.97%], test:[0.4762, 75.25%], best_acc=76.51%
epoch=006, cost=17.44, train:[0.4405, 78.97%], valid:[0.4538, 77.15%], test:[0.4359, 78.62%], best_acc=78.62%
epoch=007, cost=17.80, train:[0.4207, 81.08%], valid:[0.4594, 77.60%], test:[0.4351, 78.76%], best_acc=78.76%
epoch=008, cost=17.49, train:[0.4069, 81.51%], valid:[0.5590, 73.69%], test:[0.5284, 77.92%], best_acc=78.76%
epoch=009, cost=17.37, train:[0.4241, 80.55%], valid:[0.4403, 78.42%], test:[0.4268, 80.45%], best_acc=80.45%
epoch=010, cost=17.31, train:[0.3806, 82.68%], valid:[0.4987, 76.84%], test:[0.4757, 77.92%], best_acc=80.45%
epoch=011, cost=17.31, train:[0.3601, 83.70%], valid:[0.4834, 78.45%], test:[0.4375, 79.32%], best_acc=80.45%
epoch=012, cost=17.24, train:[0.3323, 85.49%], valid:[0.4623, 78.84%], test:[0.4230, 81.15%], best_acc=81.15%
epoch=013, cost=17.22, train:[0.2987, 87.46%], valid:[0.5213, 78.90%], test:[0.4673, 82.14%], best_acc=82.14%
epoch=014, cost=17.13, train:[0.2778, 88.21%], valid:[0.5190, 78.81%], test:[0.4576, 81.01%], best_acc=82.14%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.8620    0.7574    0.8063       371
           1     0.7662    0.8676    0.8138       340
    accuracy                         0.8101       711
   macro avg     0.8141    0.8125    0.8101       711
weighted avg     0.8162    0.8101    0.8099       711
Macro average Test Precision, Recall and F1-Score...
(0.8140984782089077, 0.8125297288726812, 0.8100529362291593, None)
Micro average Test Precision, Recall and F1-Score...
(0.810126582278481, 0.810126582278481, 0.810126582278481, None)



无句子
--------------------------------------------------
params: [start=0, batch_size=1024, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18384, 300, padding_idx=18383)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.5, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=4)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=114.21, train:[0.6853, 54.06%], valid:[0.6406, 53.24%], test:[0.6686, 53.87%], best_acc=53.87%
epoch=002, cost=117.26, train:[0.6025, 65.45%], valid:[0.5758, 70.34%], test:[0.5729, 72.01%], best_acc=72.01%
epoch=003, cost=114.66, train:[0.5347, 72.94%], valid:[0.5071, 73.80%], test:[0.5005, 73.98%], best_acc=73.98%
epoch=004, cost=115.98, train:[0.4927, 76.24%], valid:[0.4775, 75.89%], test:[0.4677, 77.50%], best_acc=77.50%
epoch=005, cost=113.33, train:[0.4529, 77.86%], valid:[0.4809, 76.45%], test:[0.4524, 78.20%], best_acc=78.20%
epoch=006, cost=115.98, train:[0.4391, 79.49%], valid:[0.4651, 76.79%], test:[0.4406, 79.04%], best_acc=79.04%
epoch=007, cost=116.90, train:[0.4197, 80.82%], valid:[0.4871, 77.41%], test:[0.4422, 79.04%], best_acc=79.04%
epoch=008, cost=115.49, train:[0.3876, 82.34%], valid:[0.5559, 74.87%], test:[0.5146, 75.67%], best_acc=79.04%
epoch=009, cost=113.64, train:[0.3957, 81.62%], valid:[0.4443, 78.14%], test:[0.4129, 81.15%], best_acc=81.15%
epoch=010, cost=113.10, train:[0.3527, 84.79%], valid:[0.4761, 78.64%], test:[0.4187, 81.86%], best_acc=81.86%
epoch=011, cost=113.58, train:[0.3240, 85.67%], valid:[0.4693, 78.76%], test:[0.4316, 81.15%], best_acc=81.86%
epoch=012, cost=115.71, train:[0.2899, 87.78%], valid:[0.5024, 77.88%], test:[0.4559, 81.43%], best_acc=81.86%
epoch=013, cost=117.03, train:[0.2774, 88.40%], valid:[0.5322, 78.76%], test:[0.4762, 80.31%], best_acc=81.86%
epoch=014, cost=113.78, train:[0.2294, 90.79%], valid:[0.6185, 78.73%], test:[0.5504, 80.17%], best_acc=81.86%
epoch=015, cost=112.68, train:[0.2159, 91.46%], valid:[0.5968, 77.74%], test:[0.5437, 80.45%], best_acc=81.86%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.8433    0.7333    0.7845       345
           1     0.7762    0.8716    0.8211       366
    accuracy                         0.8045       711
   macro avg     0.8097    0.8025    0.8028       711
weighted avg     0.8088    0.8045    0.8033       711
Macro average Test Precision, Recall and F1-Score...
(0.8097445255474454, 0.8024590163934426, 0.8028014725689145, None)
Micro average Test Precision, Recall and F1-Score...
(0.8045007032348804, 0.8045007032348804, 0.8045007032348804, None)
Process finished with exit code 0


bilstm dropout
--------------------------------------------------
params: [start=0, batch_size=1024, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18384, 300, padding_idx=18383)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.3, inplace=False)
  )
  (gcn): GraphAT(
    (dropout): Dropout(p=0.5, inplace=False)
    (conv): GATConv(300, 300, heads=3)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
    (layer_norm): LayerNorm((300,), eps=1e-06, elementwise_affine=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=17.53, train:[0.6883, 54.32%], valid:[0.7011, 53.63%], test:[0.6671, 54.15%], best_acc=54.15%
epoch=002, cost=16.36, train:[0.6452, 60.58%], valid:[0.5844, 67.47%], test:[0.5837, 70.89%], best_acc=70.89%
epoch=003, cost=16.50, train:[0.5666, 69.78%], valid:[0.5319, 73.38%], test:[0.4980, 76.65%], best_acc=76.65%
epoch=004, cost=16.36, train:[0.5213, 73.82%], valid:[0.5348, 73.35%], test:[0.5271, 74.12%], best_acc=76.65%
epoch=005, cost=16.29, train:[0.5047, 74.44%], valid:[0.4795, 76.00%], test:[0.4548, 78.76%], best_acc=78.76%
epoch=006, cost=16.21, train:[0.4754, 76.40%], valid:[0.4612, 76.42%], test:[0.4494, 79.32%], best_acc=79.32%
epoch=007, cost=16.10, train:[0.4473, 78.51%], valid:[0.4535, 77.27%], test:[0.4390, 79.47%], best_acc=79.47%
epoch=008, cost=16.08, train:[0.4342, 79.60%], valid:[0.4624, 77.72%], test:[0.4369, 80.59%], best_acc=80.59%
epoch=009, cost=16.12, train:[0.4254, 79.93%], valid:[0.4536, 78.11%], test:[0.4199, 80.45%], best_acc=80.59%
epoch=010, cost=16.12, train:[0.4118, 81.18%], valid:[0.4554, 78.39%], test:[0.4280, 79.75%], best_acc=80.59%
epoch=011, cost=16.15, train:[0.4014, 81.79%], valid:[0.4668, 78.14%], test:[0.4175, 80.31%], best_acc=80.59%
epoch=012, cost=16.10, train:[0.4088, 80.91%], valid:[0.4681, 78.11%], test:[0.4190, 79.18%], best_acc=80.59%
epoch=013, cost=16.11, train:[0.3661, 83.18%], valid:[0.4689, 77.86%], test:[0.4371, 79.89%], best_acc=80.59%
epoch=014, cost=16.18, train:[0.3684, 83.23%], valid:[0.4559, 78.76%], test:[0.4149, 80.03%], best_acc=80.59%
epoch=015, cost=16.09, train:[0.3476, 84.54%], valid:[0.4394, 79.15%], test:[0.3960, 81.29%], best_acc=81.29%
epoch=016, cost=16.17, train:[0.3342, 85.32%], valid:[0.4680, 79.60%], test:[0.4032, 80.87%], best_acc=81.29%
epoch=017, cost=16.09, train:[0.3239, 85.81%], valid:[0.4737, 79.23%], test:[0.4360, 80.03%], best_acc=81.29%
epoch=018, cost=16.02, train:[0.3043, 86.18%], valid:[0.4613, 79.26%], test:[0.3925, 80.87%], best_acc=81.29%
epoch=019, cost=16.34, train:[0.3027, 86.85%], valid:[0.4599, 79.32%], test:[0.3959, 81.15%], best_acc=81.29%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.7906    0.8483    0.8184       356
           1     0.8359    0.7746    0.8041       355
    accuracy                         0.8115       711
   macro avg     0.8132    0.8115    0.8113       711
weighted avg     0.8132    0.8115    0.8113       711
Macro average Test Precision, Recall and F1-Score...
(0.8132210888142715, 0.8114812470327584, 0.8112608757666524, None)
Micro average Test Precision, Recall and F1-Score...
(0.8115330520393812, 0.8115330520393812, 0.8115330520393812, None)
Process finished with exit code 0


--------------------------------------------------
params: [start=0, batch_size=1024, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18384, 300, padding_idx=18383)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.3, inplace=False)
  )
  (gcn): GraphAT(
    (dropout): Dropout(p=0.5, inplace=False)
    (conv): GATConv(300, 300, heads=3)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
    (layer_norm): LayerNorm((300,), eps=1e-06, elementwise_affine=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=17.89, train:[0.6815, 54.79%], valid:[0.6636, 64.43%], test:[0.6402, 67.09%], best_acc=67.09%
epoch=002, cost=16.28, train:[0.5984, 67.22%], valid:[0.5303, 72.23%], test:[0.5019, 75.11%], best_acc=75.11%
epoch=003, cost=16.36, train:[0.5398, 72.47%], valid:[0.5189, 73.44%], test:[0.4931, 74.12%], best_acc=75.11%
epoch=004, cost=16.57, train:[0.5157, 73.57%], valid:[0.4800, 75.13%], test:[0.4593, 77.64%], best_acc=77.64%
epoch=005, cost=16.33, train:[0.4936, 76.07%], valid:[0.4929, 75.07%], test:[0.4434, 80.45%], best_acc=80.45%
epoch=006, cost=16.10, train:[0.4875, 76.49%], valid:[0.4706, 76.87%], test:[0.4464, 77.78%], best_acc=80.45%
epoch=007, cost=16.56, train:[0.4615, 77.88%], valid:[0.4634, 77.49%], test:[0.4430, 78.90%], best_acc=80.45%
epoch=008, cost=16.16, train:[0.4598, 77.76%], valid:[0.4731, 76.70%], test:[0.4198, 79.61%], best_acc=80.45%
epoch=009, cost=16.11, train:[0.4403, 79.01%], valid:[0.4729, 77.04%], test:[0.4079, 80.73%], best_acc=80.73%
epoch=010, cost=16.52, train:[0.4364, 79.88%], valid:[0.4416, 78.45%], test:[0.4071, 82.14%], best_acc=82.14%
epoch=011, cost=16.29, train:[0.4034, 81.49%], valid:[0.4526, 78.62%], test:[0.4239, 80.45%], best_acc=82.14%
epoch=012, cost=16.32, train:[0.4043, 81.91%], valid:[0.4414, 79.01%], test:[0.4057, 82.28%], best_acc=82.28%
epoch=013, cost=16.23, train:[0.4013, 82.10%], valid:[0.4493, 79.60%], test:[0.4022, 82.14%], best_acc=82.28%
epoch=014, cost=16.25, train:[0.3753, 83.70%], valid:[0.4563, 78.64%], test:[0.4318, 79.47%], best_acc=82.28%
epoch=015, cost=16.12, train:[0.3597, 83.45%], valid:[0.4544, 79.46%], test:[0.4240, 81.01%], best_acc=82.28%
epoch=016, cost=16.19, train:[0.3424, 85.02%], valid:[0.4948, 79.26%], test:[0.4011, 81.86%], best_acc=82.28%
epoch=017, cost=16.26, train:[0.3254, 85.93%], valid:[0.4528, 79.40%], test:[0.4393, 81.43%], best_acc=82.28%
epoch=018, cost=16.35, train:[0.3134, 86.15%], valid:[0.4739, 79.46%], test:[0.4120, 81.72%], best_acc=82.28%
epoch=019, cost=16.23, train:[0.3090, 86.09%], valid:[0.4743, 79.46%], test:[0.4173, 81.86%], best_acc=82.28%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.8324    0.8167    0.8245       371
           1     0.8040    0.8206    0.8122       340
    accuracy                         0.8186       711
   macro avg     0.8182    0.8186    0.8184       711
weighted avg     0.8188    0.8186    0.8186       711
Macro average Test Precision, Recall and F1-Score...
(0.8182260822750737, 0.8186499127953069, 0.8183584350770876, None)
Micro average Test Precision, Recall and F1-Score...
(0.8185654008438819, 0.8185654008438819, 0.8185654008438819, None)
Process finished with exit code 0



--------------------------------------------------
params: [start=0, batch_size=1024, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18384, 300, padding_idx=18383)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.3, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=3)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
    (layer_norm): LayerNorm((300,), eps=1e-06, elementwise_affine=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=18.93, train:[0.6997, 53.04%], valid:[0.6955, 51.91%], test:[0.6856, 50.91%], best_acc=50.91%
epoch=002, cost=16.36, train:[0.6441, 61.59%], valid:[0.5763, 68.68%], test:[0.5637, 69.90%], best_acc=69.90%
epoch=003, cost=17.81, train:[0.5656, 70.47%], valid:[0.5308, 73.78%], test:[0.5121, 74.96%], best_acc=74.96%
epoch=004, cost=17.94, train:[0.5355, 72.61%], valid:[0.5162, 73.64%], test:[0.5108, 74.26%], best_acc=74.96%
epoch=005, cost=17.59, train:[0.4997, 75.16%], valid:[0.4819, 76.28%], test:[0.4735, 77.36%], best_acc=77.36%
epoch=006, cost=17.03, train:[0.4845, 76.21%], valid:[0.4617, 76.48%], test:[0.4525, 78.06%], best_acc=78.06%
epoch=007, cost=17.18, train:[0.4693, 77.32%], valid:[0.4729, 77.12%], test:[0.4467, 78.62%], best_acc=78.62%
epoch=008, cost=17.09, train:[0.4490, 78.22%], valid:[0.4462, 77.74%], test:[0.4330, 79.89%], best_acc=79.89%
epoch=009, cost=17.52, train:[0.4281, 79.49%], valid:[0.4731, 77.55%], test:[0.4368, 79.47%], best_acc=79.89%
epoch=010, cost=17.50, train:[0.4236, 80.33%], valid:[0.4419, 78.31%], test:[0.4234, 80.31%], best_acc=80.31%
epoch=011, cost=17.38, train:[0.4135, 81.02%], valid:[0.4488, 78.11%], test:[0.4331, 79.61%], best_acc=80.31%
epoch=012, cost=17.51, train:[0.4074, 81.49%], valid:[0.4547, 78.70%], test:[0.4204, 80.31%], best_acc=80.31%
epoch=013, cost=17.07, train:[0.3816, 82.70%], valid:[0.4516, 78.59%], test:[0.3958, 81.86%], best_acc=81.86%
epoch=014, cost=17.06, train:[0.3709, 82.77%], valid:[0.4432, 79.15%], test:[0.4077, 80.87%], best_acc=81.86%
epoch=015, cost=16.94, train:[0.3407, 84.52%], valid:[0.4568, 79.29%], test:[0.4078, 81.29%], best_acc=81.86%
epoch=016, cost=16.97, train:[0.3489, 85.13%], valid:[0.4476, 78.62%], test:[0.4109, 83.12%], best_acc=83.12%
epoch=017, cost=16.81, train:[0.3333, 85.24%], valid:[0.4879, 78.76%], test:[0.4251, 82.42%], best_acc=83.12%
epoch=018, cost=16.73, train:[0.3434, 84.62%], valid:[0.5089, 77.15%], test:[0.4541, 80.17%], best_acc=83.12%
epoch=019, cost=17.28, train:[0.3390, 85.21%], valid:[0.4598, 79.85%], test:[0.4094, 82.00%], best_acc=83.12%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.8529    0.7880    0.8192       368
           1     0.7898    0.8542    0.8207       343
    accuracy                         0.8200       711
   macro avg     0.8213    0.8211    0.8200       711
weighted avg     0.8225    0.8200    0.8199       711
Macro average Test Precision, Recall and F1-Score...
(0.821349294434755, 0.8211354417543415, 0.8199686654322746, None)
Micro average Test Precision, Recall and F1-Score...
(0.819971870604782, 0.819971870604782, 0.819971870604782, None)




--------------------------------------------------
params: [start=0, batch_size=512, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(7689, 300, padding_idx=7688)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.2, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=3)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=8, bias=True)
    )
  )
)
--------------------------------------------------
R8
epoch=001, cost=19.21, train:[1.1879, 59.23%], valid:[0.8348, 78.57%], test:[0.6370, 74.45%], best_acc=74.45%
epoch=002, cost=18.00, train:[0.5410, 80.78%], valid:[0.3443, 87.94%], test:[0.3986, 86.86%], best_acc=86.86%
epoch=003, cost=17.62, train:[0.3399, 88.41%], valid:[0.1794, 93.47%], test:[0.2907, 92.70%], best_acc=92.70%
epoch=004, cost=17.71, train:[0.2050, 92.59%], valid:[0.1522, 95.66%], test:[0.1610, 95.44%], best_acc=95.44%
epoch=005, cost=17.94, train:[0.1442, 94.90%], valid:[0.1191, 95.80%], test:[0.1411, 95.80%], best_acc=95.80%
epoch=006, cost=17.98, train:[0.1185, 96.17%], valid:[0.1259, 96.44%], test:[0.0964, 96.53%], best_acc=96.53%
epoch=007, cost=17.86, train:[0.0958, 96.76%], valid:[0.1016, 96.57%], test:[0.0591, 97.45%], best_acc=97.45%
epoch=008, cost=17.67, train:[0.0839, 97.27%], valid:[0.0948, 96.98%], test:[0.0842, 97.08%], best_acc=97.45%
epoch=009, cost=17.78, train:[0.0721, 97.77%], valid:[0.1136, 96.57%], test:[0.0865, 97.08%], best_acc=97.45%
epoch=010, cost=17.97, train:[0.0653, 97.61%], valid:[0.1008, 96.35%], test:[0.0788, 97.81%], best_acc=97.81%
epoch=011, cost=18.39, train:[0.0536, 98.30%], valid:[0.1189, 96.76%], test:[0.0686, 97.26%], best_acc=97.81%
epoch=012, cost=18.60, train:[0.0468, 98.50%], valid:[0.1512, 96.53%], test:[0.1143, 95.80%], best_acc=97.81%
epoch=013, cost=17.65, train:[0.0466, 98.40%], valid:[0.1242, 96.76%], test:[0.1201, 97.45%], best_acc=97.81%
epoch=014, cost=17.65, train:[0.0446, 98.46%], valid:[0.1005, 97.35%], test:[0.0456, 97.99%], best_acc=97.99%
epoch=015, cost=17.47, train:[0.0405, 98.80%], valid:[0.0995, 97.35%], test:[0.0654, 97.63%], best_acc=97.99%
epoch=016, cost=17.48, train:[0.0488, 98.34%], valid:[0.0775, 97.49%], test:[0.0345, 97.81%], best_acc=97.99%
epoch=017, cost=17.51, train:[0.0359, 98.85%], valid:[0.1968, 95.48%], test:[0.0674, 97.26%], best_acc=97.99%
epoch=018, cost=17.60, train:[0.0365, 98.85%], valid:[0.1194, 96.30%], test:[0.0429, 97.63%], best_acc=97.99%
epoch=019, cost=17.60, train:[0.0328, 98.89%], valid:[0.1036, 97.08%], test:[0.1383, 97.99%], best_acc=97.99%
epoch=020, cost=17.59, train:[0.0234, 99.17%], valid:[0.0960, 97.21%], test:[0.0490, 98.18%], best_acc=98.18%
epoch=021, cost=17.54, train:[0.0230, 99.17%], valid:[0.1456, 96.67%], test:[0.0996, 96.72%], best_acc=98.18%
epoch=022, cost=17.68, train:[0.0205, 99.39%], valid:[0.1357, 97.08%], test:[0.0717, 97.26%], best_acc=98.18%
epoch=023, cost=17.48, train:[0.0226, 99.23%], valid:[0.1262, 96.62%], test:[0.2084, 97.45%], best_acc=98.18%
epoch=024, cost=17.52, train:[0.0166, 99.37%], valid:[0.1505, 96.76%], test:[0.0613, 97.08%], best_acc=98.18%
epoch=025, cost=17.57, train:[0.0188, 99.31%], valid:[0.1336, 96.94%], test:[0.0768, 98.18%], best_acc=98.18%
epoch=026, cost=17.49, train:[0.0274, 99.21%], valid:[0.1458, 96.39%], test:[0.1884, 97.08%], best_acc=98.18%
epoch=027, cost=17.54, train:[0.0224, 99.29%], valid:[0.0958, 96.85%], test:[0.0388, 98.36%], best_acc=98.36%
epoch=028, cost=17.58, train:[0.0272, 98.99%], valid:[0.1103, 97.03%], test:[0.0709, 97.99%], best_acc=98.36%
epoch=029, cost=17.52, train:[0.0204, 99.29%], valid:[0.1102, 97.40%], test:[0.0635, 97.99%], best_acc=98.36%
epoch=030, cost=17.53, train:[0.0166, 99.45%], valid:[0.1003, 97.40%], test:[0.0632, 98.36%], best_acc=98.36%
epoch=031, cost=17.56, train:[0.0144, 99.57%], valid:[0.1220, 97.08%], test:[0.1122, 97.81%], best_acc=98.36%
epoch=032, cost=17.54, train:[0.0246, 99.23%], valid:[0.0931, 97.49%], test:[0.0841, 98.18%], best_acc=98.36%
epoch=033, cost=17.62, train:[0.0280, 99.15%], valid:[0.1265, 96.62%], test:[0.7387, 97.26%], best_acc=98.36%
epoch=034, cost=17.56, train:[0.0196, 99.39%], valid:[0.1420, 96.62%], test:[0.2416, 97.63%], best_acc=98.36%
epoch=035, cost=17.49, train:[0.0177, 99.43%], valid:[0.0865, 97.53%], test:[0.1742, 97.63%], best_acc=98.36%
epoch=036, cost=17.47, train:[0.0219, 99.43%], valid:[0.1130, 97.30%], test:[0.0673, 97.81%], best_acc=98.36%
epoch=037, cost=17.56, train:[0.0152, 99.55%], valid:[0.1067, 96.94%], test:[0.0784, 97.26%], best_acc=98.36%
epoch=038, cost=17.51, train:[0.0197, 99.29%], valid:[0.1019, 97.12%], test:[0.0459, 97.63%], best_acc=98.36%
epoch=039, cost=17.60, train:[0.0144, 99.55%], valid:[0.1156, 97.35%], test:[0.0794, 98.18%], best_acc=98.36%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     1.0000    0.9130    0.9545        23
           1     0.9167    1.0000    0.9565        22
           2     0.6667    0.6667    0.6667         3
           3     0.9821    0.9880    0.9851       167
           4     1.0000    1.0000    1.0000        21
           5     1.0000    0.9630    0.9811        27
           6     0.8667    0.8667    0.8667        15
           7     0.9926    0.9926    0.9926       270
    accuracy                         0.9818       548
   macro avg     0.9281    0.9237    0.9254       548
weighted avg     0.9821    0.9818    0.9817       548
Macro average Test Precision, Recall and F1-Score...
(0.9280919312169311, 0.9237445399056958, 0.925399977742398, None)
Micro average Test Precision, Recall and F1-Score...
(0.9817518248175182, 0.9817518248175182, 0.9817518248175182, None)


--------------------------------------------------
params: [start=0, batch_size=544, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(7689, 300, padding_idx=7688)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.2, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=3)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=8, bias=True)
    )
  )
)
--------------------------------------------------
R8
epoch=001, cost=18.79, train:[1.1201, 59.33%], valid:[0.9599, 78.80%], test:[0.7015, 75.91%], best_acc=75.91%
epoch=002, cost=17.30, train:[0.5510, 80.41%], valid:[0.3360, 87.85%], test:[0.5836, 88.14%], best_acc=88.14%
epoch=003, cost=17.43, train:[0.3470, 87.99%], valid:[0.1974, 92.55%], test:[0.1323, 93.80%], best_acc=93.80%
epoch=004, cost=17.31, train:[0.2449, 91.76%], valid:[0.1221, 94.70%], test:[0.1243, 94.34%], best_acc=94.34%
epoch=005, cost=17.33, train:[0.1640, 94.33%], valid:[0.0900, 96.44%], test:[0.1064, 96.17%], best_acc=96.17%
epoch=006, cost=17.23, train:[0.1209, 95.58%], valid:[0.1038, 95.61%], test:[0.1737, 94.89%], best_acc=96.17%
epoch=007, cost=17.37, train:[0.0999, 96.23%], valid:[0.0937, 96.76%], test:[0.0681, 96.35%], best_acc=96.35%
epoch=008, cost=17.21, train:[0.1032, 97.00%], valid:[0.0901, 96.67%], test:[0.0672, 96.17%], best_acc=96.35%
epoch=009, cost=17.23, train:[0.1071, 96.72%], valid:[0.0959, 96.48%], test:[0.1066, 97.08%], best_acc=97.08%
epoch=010, cost=17.38, train:[0.0922, 97.04%], valid:[0.1099, 95.93%], test:[0.0701, 95.62%], best_acc=97.08%
epoch=011, cost=17.24, train:[0.0813, 97.43%], valid:[0.0849, 96.62%], test:[0.1338, 96.53%], best_acc=97.08%
epoch=012, cost=17.33, train:[0.0680, 97.49%], valid:[0.1403, 96.39%], test:[0.0676, 96.53%], best_acc=97.08%
epoch=013, cost=17.58, train:[0.0531, 97.99%], valid:[0.1523, 96.48%], test:[0.1553, 95.99%], best_acc=97.08%
epoch=014, cost=17.32, train:[0.0509, 98.22%], valid:[0.0658, 97.58%], test:[0.0582, 97.08%], best_acc=97.08%
epoch=015, cost=17.30, train:[0.0384, 98.66%], valid:[0.0727, 97.35%], test:[0.0567, 97.63%], best_acc=97.63%
epoch=016, cost=17.22, train:[0.0451, 98.30%], valid:[0.1242, 95.66%], test:[0.0666, 96.17%], best_acc=97.63%
epoch=017, cost=17.26, train:[0.0934, 96.86%], valid:[0.0960, 96.53%], test:[0.0623, 96.35%], best_acc=97.63%
epoch=018, cost=17.27, train:[0.0718, 97.39%], valid:[0.0846, 97.17%], test:[0.0457, 97.81%], best_acc=97.81%
epoch=019, cost=17.35, train:[0.0410, 98.62%], valid:[0.1085, 96.94%], test:[0.0862, 97.63%], best_acc=97.81%
epoch=020, cost=17.25, train:[0.0306, 98.97%], valid:[0.0792, 97.58%], test:[0.0438, 97.99%], best_acc=97.99%
epoch=021, cost=17.19, train:[0.0245, 99.15%], valid:[0.1095, 97.53%], test:[0.0486, 97.45%], best_acc=97.99%
epoch=022, cost=17.17, train:[0.0192, 99.31%], valid:[0.0817, 97.35%], test:[0.0640, 97.63%], best_acc=97.99%
epoch=023, cost=17.28, train:[0.0176, 99.33%], valid:[0.0928, 97.08%], test:[0.0521, 97.81%], best_acc=97.99%
epoch=024, cost=17.30, train:[0.0156, 99.43%], valid:[0.2273, 97.30%], test:[0.0516, 97.63%], best_acc=97.99%
epoch=025, cost=17.47, train:[0.0152, 99.53%], valid:[0.0895, 97.40%], test:[0.0596, 97.63%], best_acc=97.99%
epoch=026, cost=17.23, train:[0.0532, 99.15%], valid:[0.1336, 96.71%], test:[0.1287, 95.62%], best_acc=97.99%
epoch=027, cost=17.26, train:[0.1299, 95.75%], valid:[0.1382, 95.57%], test:[0.3197, 96.53%], best_acc=97.99%
epoch=028, cost=17.20, train:[0.0730, 97.59%], valid:[0.0984, 96.21%], test:[0.0508, 97.45%], best_acc=97.99%
epoch=029, cost=17.24, train:[0.0406, 98.54%], valid:[0.1311, 96.98%], test:[0.0415, 98.72%], best_acc=98.72%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.9565    0.9565    0.9565        23
           1     0.9889    0.9963    0.9926       269
           2     1.0000    1.0000    1.0000        11
           3     1.0000    0.9565    0.9778        23
           4     0.8889    0.8000    0.8421        10
           5     0.9885    0.9885    0.9885       174
           6     1.0000    1.0000    1.0000        36
           7     1.0000    1.0000    1.0000         2
    accuracy                         0.9872       548
   macro avg     0.9779    0.9622    0.9697       548
weighted avg     0.9871    0.9872    0.9871       548
Macro average Test Precision, Recall and F1-Score...
(0.9778557830555816, 0.9622289691585434, 0.9696878899731421, None)
Micro average Test Precision, Recall and F1-Score...
(0.9872262773722628, 0.9872262773722628, 0.9872262773722628, None)


--------------------------------------------------
params: [start=0, batch_size=560, lr=0.001, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(7689, 300, padding_idx=7688)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.2, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=3)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=8, bias=True)
    )
  )
)
--------------------------------------------------
R8
epoch=001, cost=436.14, train:[1.2787, 63.34%], valid:[0.5671, 80.72%], test:[0.7613, 79.56%], best_acc=79.56%
epoch=002, cost=537.61, train:[0.6547, 82.18%], valid:[0.4040, 88.12%], test:[0.4233, 88.87%], best_acc=88.87%
epoch=003, cost=450.68, train:[0.3951, 88.82%], valid:[0.2214, 92.96%], test:[0.2089, 94.89%], best_acc=94.89%
epoch=004, cost=433.18, train:[0.2271, 93.60%], valid:[0.1717, 95.34%], test:[0.1629, 94.34%], best_acc=94.89%
epoch=005, cost=453.22, train:[0.1812, 94.63%], valid:[0.1436, 95.89%], test:[0.1092, 96.53%], best_acc=96.53%
epoch=006, cost=454.00, train:[0.1332, 95.95%], valid:[0.1270, 96.35%], test:[0.1105, 96.17%], best_acc=96.53%
epoch=007, cost=449.24, train:[0.1201, 96.37%], valid:[0.1334, 96.16%], test:[0.0894, 97.45%], best_acc=97.45%
epoch=008, cost=448.69, train:[0.0928, 97.31%], valid:[0.1407, 96.35%], test:[0.0876, 97.26%], best_acc=97.45%
epoch=009, cost=449.90, train:[0.0814, 97.57%], valid:[0.1829, 95.11%], test:[0.1032, 96.17%], best_acc=97.45%
epoch=010, cost=444.79, train:[0.0763, 97.91%], valid:[0.1115, 96.89%], test:[0.0825, 96.53%], best_acc=97.45%
epoch=011, cost=451.81, train:[0.0599, 98.42%], valid:[0.1465, 96.71%], test:[0.1559, 95.99%], best_acc=97.45%
epoch=012, cost=448.49, train:[0.0640, 98.12%], valid:[0.0988, 97.30%], test:[0.0797, 97.63%], best_acc=97.63%
epoch=013, cost=407.44, train:[0.0481, 98.44%], valid:[0.1215, 97.08%], test:[0.0670, 97.63%], best_acc=97.63%
epoch=014, cost=403.77, train:[0.0445, 98.48%], valid:[0.1424, 96.62%], test:[0.0856, 97.81%], best_acc=97.81%
epoch=015, cost=408.00, train:[0.0381, 98.89%], valid:[0.1483, 96.71%], test:[0.0723, 97.81%], best_acc=97.81%
epoch=016, cost=410.16, train:[0.0306, 99.03%], valid:[0.1750, 95.84%], test:[0.0728, 97.26%], best_acc=97.81%
epoch=017, cost=408.57, train:[0.0317, 99.11%], valid:[0.1423, 96.03%], test:[0.0773, 97.81%], best_acc=97.81%
epoch=018, cost=405.02, train:[0.0345, 98.89%], valid:[0.1330, 96.89%], test:[0.0959, 97.45%], best_acc=97.81%
epoch=019, cost=410.58, train:[0.0424, 98.80%], valid:[0.1182, 96.89%], test:[0.0595, 98.36%], best_acc=98.36%
epoch=020, cost=413.40, train:[0.0413, 98.83%], valid:[0.1254, 97.30%], test:[0.1341, 97.26%], best_acc=98.36%
epoch=021, cost=409.85, train:[0.0442, 98.56%], valid:[0.1316, 97.03%], test:[0.1154, 97.63%], best_acc=98.36%
epoch=022, cost=400.66, train:[0.0368, 99.11%], valid:[0.1143, 97.26%], test:[0.0931, 97.45%], best_acc=98.36%
epoch=023, cost=410.13, train:[0.0195, 99.45%], valid:[0.1319, 96.98%], test:[0.0742, 98.18%], best_acc=98.36%
epoch=024, cost=412.74, train:[0.0213, 99.35%], valid:[0.1185, 97.53%], test:[0.0595, 98.54%], best_acc=98.54%
epoch=025, cost=410.22, train:[0.0212, 99.39%], valid:[0.1564, 96.80%], test:[0.0598, 98.18%], best_acc=98.54%
epoch=026, cost=403.01, train:[0.0403, 98.83%], valid:[0.1437, 96.57%], test:[0.1235, 96.17%], best_acc=98.54%
epoch=027, cost=397.50, train:[0.1031, 96.46%], valid:[0.1299, 96.94%], test:[0.1049, 96.72%], best_acc=98.54%
epoch=028, cost=404.86, train:[0.0457, 98.56%], valid:[0.1069, 97.35%], test:[0.0780, 97.99%], best_acc=98.54%
epoch=029, cost=403.86, train:[0.0285, 99.15%], valid:[0.1495, 96.76%], test:[0.0773, 97.63%], best_acc=98.54%
epoch=030, cost=417.93, train:[0.0290, 99.17%], valid:[0.1812, 96.12%], test:[0.0900, 96.90%], best_acc=98.54%
epoch=031, cost=417.09, train:[0.0259, 99.19%], valid:[0.1538, 96.57%], test:[0.0809, 97.63%], best_acc=98.54%
epoch=032, cost=406.44, train:[0.0198, 99.49%], valid:[0.1561, 96.48%], test:[0.0878, 97.63%], best_acc=98.54%
epoch=033, cost=417.13, train:[0.0180, 99.45%], valid:[0.1313, 96.98%], test:[0.0825, 97.81%], best_acc=98.54%
epoch=034, cost=408.55, train:[0.0179, 99.62%], valid:[0.1372, 97.08%], test:[0.0973, 97.99%], best_acc=98.54%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.7692    1.0000    0.8696        10
           1     0.9967    0.9839    0.9903       311
           2     1.0000    0.9474    0.9730        19
           3     1.0000    0.8000    0.8889         5
           4     0.9643    1.0000    0.9818       135
           5     1.0000    0.9583    0.9787        24
           6     0.9565    0.9565    0.9565        23
           7     0.9500    0.9048    0.9268        21
    accuracy                         0.9799       548
   macro avg     0.9546    0.9439    0.9457       548
weighted avg     0.9814    0.9799    0.9801       548
Macro average Test Precision, Recall and F1-Score...
(0.9545976117070862, 0.9438635284825373, 0.9457013668607135, None)
Micro average Test Precision, Recall and F1-Score...
(0.9799270072992701, 0.9799270072992701, 0.9799270072992701, None)
Process finished with exit code 0



3090 mix
--------------------------------------------------
params: [start=0, batch_size=1024, lr=0.0003, weight_decay=0.0005]
--------------------------------------------------
Model(
  (embed): Embedding(18384, 300, padding_idx=18383)
  (bilstm): BiLstm(
    (lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
    (dropout_layer): Dropout(p=0.4, inplace=False)
  )
  (gcn): GraphAT(
    (conv): GATConv(300, 300, heads=3)
    (encode): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=300, bias=True)
    )
    (gru): GRUUint_s(
      (lin_z0): Linear(in_features=300, out_features=300, bias=True)
      (lin_z1): Linear(in_features=300, out_features=300, bias=True)
      (lin_z2): Linear(in_features=300, out_features=300, bias=True)
      (lin_r0): Linear(in_features=300, out_features=300, bias=True)
      (lin_r1): Linear(in_features=300, out_features=300, bias=True)
      (lin_r2): Linear(in_features=300, out_features=300, bias=True)
      (lin_h0): Linear(in_features=300, out_features=300, bias=True)
      (lin_h1): Linear(in_features=300, out_features=300, bias=True)
      (lin_h2): Linear(in_features=300, out_features=300, bias=True)
    )
    (W2): Linear(in_features=300, out_features=300, bias=True)
    (w2): Linear(in_features=300, out_features=1, bias=True)
  )
  (read): ReadoutLayer(
    (att): Linear(in_features=300, out_features=1, bias=True)
    (emb): Linear(in_features=300, out_features=300, bias=True)
    (mlp): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=300, out_features=2, bias=True)
    )
  )
)
--------------------------------------------------
mr
epoch=001, cost=28.28, train:[4.2708, 61.42%], valid:[1.7529, 72.48%], test:[0.5092, 74.82%], best_acc=74.82%
epoch=002, cost=26.39, train:[3.5664, 72.86%], valid:[1.6256, 74.73%], test:[0.4908, 76.09%], best_acc=76.09%
epoch=003, cost=26.65, train:[3.3645, 74.14%], valid:[1.5751, 76.17%], test:[0.4827, 76.09%], best_acc=76.09%
epoch=004, cost=26.58, train:[3.1852, 76.29%], valid:[1.5390, 76.48%], test:[0.4568, 78.20%], best_acc=78.20%
epoch=005, cost=26.71, train:[3.0991, 76.86%], valid:[1.4896, 77.49%], test:[0.4444, 79.61%], best_acc=79.61%
epoch=006, cost=26.73, train:[2.9479, 78.87%], valid:[1.4558, 77.97%], test:[0.4382, 79.89%], best_acc=79.89%
epoch=007, cost=26.50, train:[2.8231, 80.02%], valid:[1.4408, 78.98%], test:[0.4461, 79.32%], best_acc=79.89%
epoch=008, cost=26.59, train:[2.7718, 80.85%], valid:[1.4580, 78.33%], test:[0.4436, 78.62%], best_acc=79.89%
epoch=009, cost=26.62, train:[2.6341, 81.35%], valid:[1.4233, 79.04%], test:[0.4323, 80.59%], best_acc=80.59%
epoch=010, cost=26.67, train:[2.5726, 82.43%], valid:[1.4879, 78.36%], test:[0.4541, 80.03%], best_acc=80.59%
epoch=011, cost=26.52, train:[2.4705, 83.26%], valid:[1.4357, 79.21%], test:[0.4262, 80.31%], best_acc=80.59%
epoch=012, cost=25.49, train:[2.3481, 84.24%], valid:[1.4684, 79.12%], test:[0.4463, 80.45%], best_acc=80.59%
epoch=013, cost=24.19, train:[2.2489, 85.04%], valid:[1.4908, 78.17%], test:[0.4537, 78.76%], best_acc=80.59%
epoch=014, cost=24.26, train:[2.1457, 85.56%], valid:[1.4947, 79.43%], test:[0.4533, 80.87%], best_acc=80.87%
Test Precision, Recall and F1-Score...
              precision    recall  f1-score   support
           0     0.7853    0.8475    0.8152       354
           1     0.8359    0.7703    0.8017       357
    accuracy                         0.8087       711
   macro avg     0.8106    0.8089    0.8085       711
weighted avg     0.8107    0.8087    0.8085       711
Macro average Test Precision, Recall and F1-Score...
(0.810603287767151, 0.8088828751839718, 0.808483331220687, None)
Micro average Test Precision, Recall and F1-Score...
(0.8087201125175809, 0.8087201125175809, 0.8087201125175809, None)
