#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

from sklearn.feature_selection import RFECV
# def make_predictions(tr_df, tt_df, feature_columns, target, lgb_params, NFOLDS=2):
#     folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
#
#     X, y = tr_df[feature_columns], tr_df[target]
#     P, P_y = tt_df[feature_columns], tt_df[target]
#
#     tt_df = tt_df[['TransactionID', target]]
#     predictions = np.zeros(len(tt_df))
#
#     for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#         print('Fold:', fold_)
#         tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
#         vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
#
#         print(len(tr_x), len(vl_x))
#         tr_data = lgb.Dataset(tr_x, label=tr_y)
#
#         if LOCAL_TEST:
#             vl_data = lgb.Dataset(P, label=P_y)
#         else:
#             vl_data = lgb.Dataset(vl_x, label=vl_y)
#
#         estimator = lgb.train(
#             lgb_params,
#             tr_data,
#             valid_sets=[tr_data, vl_data],
#             verbose_eval=True,
#         )
#
#         pp_p = estimator.predict(P)
#         predictions += pp_p / NFOLDS
#
#         if LOCAL_TEST:
#             feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), X.columns)),
#                                        columns=['Value', 'Feature'])
#             print(feature_imp)
#
#         del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
#         gc.collect()
#
#     tt_df['prediction'] = predictions
#
#     return tt_df
#
#
# # _________________________________________________________________________________________________
# # train model:
# if LOCAL_TEST:
#     test_predictions = make_predictions(train_trans, test_trans, feat_columns, TARGET, lgb_params)
#     print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
# else:
#     lgb_params['learning_rate'] = 0.01
#     lgb_params['n_estimators'] = 800
#     lgb_params['early_stopping_rounds'] = 100
#     test_predictions = make_predictions(train_trans, test_trans, feat_columns, TARGET, lgb_params, NFOLDS=2)
#
# # _________________________________________________________________________________________________
# # export:
# if not LOCAL_TEST:
#     test_predictions['isFraud'] = test_predictions['prediction']
#     test_predictions[['TransactionID','isFraud']].to_csv('submission.csv', index=False)


# # _______________________________________________________________________________________
# # set encoding for each feature & do encoding:
# FOE = FEDesc
# trans_feat_op_table = [
#     # FOE('card4', FreqEnc(count_on=pd.concat([train_trans['card4'], test_trans['card4']])), ),
#     # FOE('card6', FreqEnc(count_on=pd.concat([train_trans['card6'], test_trans['card6']])), ),
#     # FOE('ProductCD', FreqEnc(count_on=pd.concat([train_trans['ProductCD'], test_trans['ProductCD']])), ),
#     FOE('M1', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('M2', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('M3', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('M4', RegexEnc(r'M(\d*)')),
#     FOE('M5', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('M6', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('M7', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('M8', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('M9', MapEnc({'T': 1, 'F': 0}), ),
# ]
#
# id_feat_op_table = [
#     FOE('id_12', MapEnc({'Found': 1, 'NotFound': 0}), ),
#
#     FOE('id_15', MapEnc({'New': 2, 'Found': 1, 'Unknown': 0}), ),
#     FOE('id_16', MapEnc({'Found': 1, 'NotFound': 0}), ),
#
#     FOE('id_23', MapEnc({'TRANSPARENT': 4, 'IP_PROXY': 3, 'IP_PROXY:ANONYMOUS': 2, 'IP_PROXY:HIDDEN': 1}), ),
#
#     FOE('id_27', MapEnc({'Found': 1, 'NotFound': 0}), ),
#     FOE('id_28', MapEnc({'New': 2, 'Found': 1}), ),
#     FOE('id_29', MapEnc({'Found': 1, 'NotFound': 0}), ),
#
#     # FOE('id_33', RegexEnc(r'(\d*)x\d*'), dst='id_33_W'),
#     # FOE('id_33', RegexEnc(r'\d*x(\d*)'), dst='id_33_H'),
#     # FOE('id_33', RemoveFeat(), ),   # delete col 'id_33'
#
#     FOE('id_34', RegexEnc(r'.*:([-+]?\d+)'), ),
#     FOE('id_35', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('id_36', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('id_37', MapEnc({'T': 1, 'F': 0}), ),
#     FOE('id_38', MapEnc({'T': 1, 'F': 0}), ),
#
#     FOE('DeviceType', MapEnc({'desktop': 1, 'mobile': 0}), ),
# ]
#
# print('-'*30 + ' Encoding train transactions...')
# train_trans_encoded = do_feat_ops(train_trans, trans_feat_op_table)
# print('-'*30 + ' Encoding test transactions...')
# test_trans_encoded = do_feat_ops(test_trans, trans_feat_op_table)
#
# print('-'*30 + ' Encoding train identity...')
# train_id_encoded = do_feat_ops(train_id, id_feat_op_table)
# print('-'*30 + ' Encoding test identity...')
# test_id_encoded = do_feat_ops(test_id, id_feat_op_table)
#


