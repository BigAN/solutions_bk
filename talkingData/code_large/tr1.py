select cvr_pv_exp_id,
       pv_num,
       click_num,
       order_num,
       bid,
       price,
       final_charge,
       (mt_butie+total) as revenue,
       click_num/pv_num as ctr,
       price/click_num as cpc,
       price/bid as chargeRate,
       price/pv_num*1000 as cpm,
       order_num /click_num as cvr,
       (mt_butie+total)/order_num as avg_price,
       (mt_butie+total)/price as roi,
       (mt_butie+total)/pv_num*1000 as gmv
  from (
        select cvr_pv_exp_id,
               count(1) as pv_num
          from (
                select get_json_object(expids, '$.AS') as pv_exp_id,
                       case when get_json_object(expids, '$.OCPC_CALB_CVR') is null then '-1'
                            else get_json_object(expids, '$.OCPC_CALB_CVR')
                             END as cvr_pv_exp_id
                  from log.dpods_baymax_wm_charge_service_cpvlog
                 where dt BETWEEN '20180427' and '$$yesterday_compact'
                   and act=3
                   and isvalid='PASS'
                   and FROM_UNIXTIME(cast (SUBSTR(log_time,1,10) as bigint),'yyyyMMdd')=dt
                   and mtdpid <> ''
                   and slot in ('1', '3')-- group by get_json_object(expids, '$.AS')

               ) inner_pv_log
         group by cvr_pv_exp_id
       ) pv_log
  left join (
        select case when get_json_object(charge_info_map['expids'], '$.OCPC_CALB_CVR') is null then '-1'
                    else get_json_object(charge_info_map['expids'], '$.OCPC_CALB_CVR')
                     END as cvr_order_exp_id,
               sum(sub_mt_ord_charge_fee) as mt_butie, -- 美团补贴金额
sum(total) as total ,-- 用户实际支付金额
count(distinct wm_order_id) as order_num -- 订单号

          from (
                select str_to_map(if(coalesce(get_json_object(entry_info,'$.fstAd.chargeInfo'),get_json_object(entry_info,'$.secAd.chargeInfo')) is null,'',coalesce(get_json_object(entry_info,'$.fstAd.chargeInfo'),get_json_object(entry_info,'$.secAd.chargeInfo'))),'&','=') as charge_info_map,
                       sub_mt_ord_charge_fee, -- 美团补贴金额
total,-- 用户实际支付金额
wm_order_id -- 订单号

                  from mart_waimai.fact_xianfu_waimai_log_server_order_ad
                 where dt BETWEEN '20180427' and '$$yesterday_compact'
                   and ad_type=2
                   and is_arrange=1 -- 表示提单

                   and cpc_entry_id in ('1', '3')
               ) t1
         where charge_info_map['expids'] is not null
           and charge_info_map['slot'] in ('1','3')-- and get_json_object(charge_info_map['expids'], '$.CTR') in ('601', '602') )

         group by case when get_json_object(charge_info_map['expids'], '$.OCPC_CALB_CVR') is null then '-1'
                       else get_json_object(charge_info_map['expids'], '$.OCPC_CALB_CVR')
                        END
       ) order_log
    on pv_log.cvr_pv_exp_id = order_log.cvr_order_exp_id
  left join (
        select case when get_json_object(expids, '$.OCPC_CALB_CVR') is null then '-1'
                    else get_json_object(expids, '$.OCPC_CALB_CVR')
                     END as cvr_click_exp_id,
               count(1) as click_num,
               sum(price)/100 as price,
               sum(final_charge)/100 as final_charge,
               sum(bid)/100 as bid
          from log.dpods_baymax_wm_reporter_service_cpvlog
         where dt BETWEEN '20180427' and '$$yesterday_compact'
           and act=2
           and isvalid='PASS'
           and FROM_UNIXTIME(cast (SUBSTR(log_time,1,10) as bigint),'yyyyMMdd')=dt
           and log_file_name='baymax-reporter-charge-log'
           and slot in ('1', '3')
         group by case when get_json_object(expids, '$.OCPC_CALB_CVR') is null then '-1'
                       else get_json_object(expids, '$.OCPC_CALB_CVR')
                        END
       ) click_log
    on pv_log.cvr_pv_exp_id = click_log.cvr_click_exp_id