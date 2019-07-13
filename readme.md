Here is a sample running command:

```
python main.py \
--name RRDB_0_4 \
--batch_size 80 \
--loss 1*RMSE+1*FaceSphere+1*GanLoss \
--niter 30000 \
--lr 1e-4 \
--pre_train_model /BIGDATA1/nsccgz_yfdu_5/lzh/IAA/SISR-pytorch-framework/experiments/RRDB_0_1/model/4000.pth
```

View `options.py` to explore more arguments.