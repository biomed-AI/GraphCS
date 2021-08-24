import pandas as pd
c0=pd.DataFrame(columns=['CEL-Seq2','Drop-seq','inDrops','Seq-Well','Smart-seq2','mouse_retina','mouse_brain'])
c0.to_csv('./cross-platforms.csv')
c0=pd.DataFrame(columns=['Baron_mouse-Baron_human','Baron_mouse-Segerstolpe','Baron_human-Baron_mouse','Baron_mouse-Combination'])
c0.to_csv('./cross-species.csv')
c0=pd.DataFrame(columns=['0.2','0.4','0.6','0.8','1','1.2','1.4','1.6'])
c0.to_csv('./simulate.csv')
c0.to_csv('./simulate-std.csv')