from pymilvus import Collection, connections
connections.connect("default", host="localhost", port="19530")
#在cfgs/config.yaml中的默认值
col_name = "history_rag" 
col = Collection(col_name)
col.load()
col.drop()