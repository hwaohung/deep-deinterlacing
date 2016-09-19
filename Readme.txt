把專案放在caffe\example底下

Train data放入Train資料夾，Test data放入Test資料夾
執行generate_data會先做成mosaicking後存成成hdf5檔案(Train跟test分兩次製作)

有了train.h5跟test.h5就可以用caffe來train了
(net指定資料是寫一個txt位置ex:train.txt test.txt，而txt中就是指定資料位置，確定是否可以直接指定資料位置)

train完後，可以用demo_Demosaicking執行。
