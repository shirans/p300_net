# from hdfs import InsecureClient
# from hdfs3 import HDFileSystem
# import pandas as pd
#
#
#
# client_hdfs = InsecureClient('http://taz-hdfs.service.il.consul.taboolasyndication.com:50070')
# with client_hdfs.read('/user-data/lookalike/stats/2020-03-30_07-21-43/features/part-00000-b3a0d786-8a57-47db-99e2-404d095e0c4e-c000.csv', encoding = 'utf-8') as reader:
#     df = pd.read_csv(reader,index_col=0)
#     print(df.tail(10))
#
#
# # # client_hdfs = InsecureClient('http://' + os.environ['IP_HDFS'] + ':50070')
# # hdfs = HDFileSystem(host='http://taz-hdfs.service.consul.taboolasyndication.com/', port=50070)
# # with hdfs.open('/user-data/lookalike/stats/2020-03-30_07-21-43/features.csv') as f:
# #     data = f.read(10)
# # import numpy as np
# # f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
# # exp = np.exp(f)
# # print("exp: " + str(exp))
# # np_sum = np.sum(exp)
# # print("np sum:" ,np_sum)
# # p = exp / np_sum  # Bad: Numeric problem, potential blowup
# # print("result:" ,p)
# # # instead: first shift the values of f so that the highest number is 0:
# # f -= np.max(f) # f becomes [-666, -333, 0]
# # np_exp = np.exp(f)
# # print("exp: " + str(np_exp))
# # np_sum = np.sum(np_exp)
# # print("np sum:" + str(np_sum))
# # p = np_exp / np_sum  # safe to do, gives the correct answer
# # print("result:" ,p)
