'''
@author: Sparkle Russell-Puleri

'''

--This script was used to create the 30 day readmission cohort for this mini project

set search_path to mimiciii;

--1. Readmission and demographic variables

DROP MATERIALIZED VIEW IF EXISTS mimiciii.readmission CASCADE;
create materialized view mimiciii.readmission as (
with 
first_discharge as(
		select subject_id, 
		       min(outtime) as first_disch
		from mimiciii.icustays
		where first_careunit <> 'NICU'
		group by subject_id),
readmit as (
        select a.*,
              round(cast((EXTRACT(DAY from (a.intime - b.first_disch)) +
                        EXTRACT(HOUR from (a.intime - b.first_disch))/24+
                        EXTRACT(MINUTE from (a.intime - b.first_disch))/60/24) as numeric),0) as days,
              (EXTRACT(DAY from (a.intime - b.first_disch)) +
                        EXTRACT(HOUR from (a.intime - b.first_disch))/24+
                        EXTRACT(MINUTE from (a.intime - b.first_disch))/60/24) as days_raw,
              ROW_NUMBER() over (partition by a.subject_id order by a.outtime, hadm_id) as events
        from mimiciii.icustays a  
        left join first_discharge b
        on a.subject_id=b.subject_id
        where first_careunit <> 'NICU'),
days_to_readmit as(
        select subject_id, days as days_elapsed
	    from readmit
	    where events=2),
first_readmit as(
		select a.* , b.days_elapsed, c.first_disch
		from readmit a
		left join days_to_readmit b 
		on a.subject_id=b.subject_id
		inner join first_discharge c 
		on a.subject_id=c.subject_id and a.outtime=c.first_disch),
demographics as (
		select subject_id, gender,
		       dob
		from mimiciii.patients),
sociodemo_operational as(
        select subject_id, hadm_id, admittime, 
               (case 
                  when discharge_location  in ('SNF', 'REHAB/DISTINCT PART HOSP', 'DISC-TRAN CANCER/CHLDRN H',
                                               'LONG TERM CARE HOSPITAL', 'DISC-TRAN TO FEDERAL HC',
                                               'HOSPICE-MEDICAL FACILITY', 'SHORT TERM HOSPITAL', 'DISCH-TRAN TO PSYCH HOSP',
                                               'HOSPICE-HOME', 'ICF', 'SNF-MEDICAID ONLY CERTIF')
                  then 'SNF'
                when discharge_location in ('HOME WITH HOME IV PROVIDR', 'HOME HEALTH CARE', 'HOME')
                  then 'HOME'
               else 'OTHER' end) as discharge_location,
               language, insurance, ethnicity, dischtime
        from mimiciii.admissions),
deathtimes as(
        select subject_id, hadm_id, deathtime, hospital_expire_flag
        from mimiciii.admissions
        where deathtime is not NULL AND hospital_expire_flag <> 1)
select a.subject_id, a.hadm_id, a.icustay_id,
       a.first_careunit, a.intime, 
       a.outtime, a.los, b.gender, b.dob,
       c.discharge_location, c.language, c.insurance, c.ethnicity,
       c.dischtime, d.deathtime, d.hospital_expire_flag,
       round(( cast(a.intime as date) - cast(b.dob as date)) / 365.242 , 2 ) as age,
       (case when a.days_elapsed is null then 0
        else a.days_elapsed end) as days_since_discharge,
       (case when (a.days_elapsed > 0) and (a.days_elapsed <30) then 1
        else 0 end) as readmission_30_days
from first_readmit a
inner join demographics b
on a.subject_id=b.subject_id
left join sociodemo_operational c
on a.subject_id=c.subject_id and a.hadm_id=c.hadm_id
left join deathtimes d 
on a.subject_id=d.subject_id
where round(( cast(a.intime as date) - cast(b.dob as date)) / 365.242 , 2 ) > 18);

select discharge_location from mimiciii.readmission;

-- 2. Comorbities:
    -- 1.  Heart Failure
    -- 2.  COPD
    -- 3.  Renal Disease
    -- 4.  Cancer (with and without metastasis)
    -- 5.  Diabetes mellitus

DROP MATERIALIZED VIEW IF EXISTS mimiciii.comorbidities_table CASCADE;
CREATE MATERIALIZED VIEW mimiciii.comorbidities_table as(
with comorbitites as(
select b.subject_id, b.hadm_id, b.icd9_code, a.long_title,
       (case when a.icd9_code in (
			  '39891', '4280', '4281', 
			  '42820', '42821', '42822', 
			  '42823', '42830', '42831', 
			  '42832', '42833', '42840', 
			  '42841', '42842', '42843', '4289') then 1
			else 0 end) as congestive_heart_failure,
	    (case when b.icd9_code in (
	          '490', '4910', '4911', '4912',
              '49120', '49121', '49122', '4918', 
              '4919', '4920', '4928', '494', '4940', 
              '4941', '496') then 1 
         else 0 end) as COPD,
        (case when lower(b.icd9_code) in (
             '5800', '5804', '58081', '58089', 
             '5809', '5810', '5811', '5812', '5813', 
             '58181', '58189', '5819', '5820', '5821', 
             '5822', '5824', '58281', '58289', '5829', 
             '5830', '5831', '5832', '5834', '5836', '5837', 
             '58381', '58389', '5839', '587') then 1
             else 0 end) as renal_failure,
        (case when lower(a.long_title) like '%malignant' then 1
              else 0 end) as cancer,
        (case when a.icd9_code in (
              '24901', '24910', '24911', '24920', '24921', 
              '24930', '24931', '24940', '24941', '24950', 
	           '24951', '24960', '24961', '24970', '24971', 
	           '24980', '24981', '24990', '24991', '25002', 
	           '25003', '25010', '25011', '25012', '25013', 
	           '25020', '25021', '25022', '25023', '25030', 
	           '25031', '25032', '25033', '25040', '25041', 
	           '25042', '25043', '25050', '25051', '25052', 
	           '25053', '25060', '25061', '25062', '25063', 
	           '25070', '25071', '25072', '25073', '25080', 
	           '25081', '25082', '25083', '25090', '25091', 
	           '25092', '25093', '24900', '25000', '25001', 
	           '7902', '79021', '79022', '79029', '7915', 
	           '7916', 'V4585', 'V5391', 'V6546') then 1
	      else 0 end) as diabetes
      from mimiciii.d_icd_diagnoses a
      left join mimiciii.diagnoses_icd b
      on a.icd9_code=b.icd9_code),
 first_dishcarge as (
		select subject_id, hadm_id,
		       row_number() over (partition by subject_id order by outtime, hadm_id)as first_disch
		from mimiciii.icustays
		where first_careunit <> 'NICU'),
codes_first_visit as (
	select a.*
	from first_dishcarge b 
	left join comorbitites a  
	on a.hadm_id=b.hadm_id and a.subject_id=b.subject_id
	where b.first_disch=1),
all_comobidities as (
	select a.subject_id, a.hadm_id,
	       max(congestive_heart_failure) over (partition by subject_id order by hadm_id) as cong_heart_fail,
	       max(COPD) over (partition by subject_id order by hadm_id) as COPD_,
	       max(renal_failure) over (partition by subject_id order by hadm_id) as renal_fail,
	       max(cancer) over (partition by subject_id order by hadm_id) as cancers,
	       max(diabetes) over (partition by subject_id order by hadm_id) as diabetes_,
	       row_number() over (partition by subject_id order by hadm_id) as rn
	from codes_first_visit a)
--select * from all_comobidities where subject_id in (3, 4, 6, 9, 21);
select a.subject_id, a.hadm_id,
       a.cong_heart_fail as congestive_heart_failure,
       a.COPD_ as COPD, a.renal_fail as renal_failure,
       a.cancers as cancer, a.diabetes_ as diabetes
from all_comobidities a
where rn=1);



--3. Labs 
DROP MATERIALIZED VIEW IF EXISTS mimiciii.timeseries_table CASCADE;
CREATE MATERIALIZED VIEW mimiciii.timeseries_table as(
with labs as 
(select a.subject_id, a.hadm_id, a.charttime, a.valuenum,
        a.charttime::date as date_entered,
        b.label, b.itemid, e.icustay_id
 from mimiciii.labevents a
 left join mimiciii.d_labitems b
 on a.itemid=b.itemid
 inner join mimiciii.readmission e on a.subject_id=e.subject_id 
 where b.itemid in (50862, 50878, 50885,
                    50963, 50809, 50971,
                    50983, 50863, 50820,
                    50821, 50818, 51275,
                    51237, 51144, 51222,
                    51265, 51300, 51002,
                    50912, 50911)
 and a.hadm_id is not null
),
chartsevents as
(select a.subject_id, a.hadm_id,a.charttime, a.valuenum,
        a.charttime::date as date_entered,
        b.label, b.itemid, a.icustay_id
 from mimiciii.chartevents a
 left join mimiciii.d_items b 
 on a.itemid=b.itemid
 inner join mimiciii.readmission c
 on a.subject_id=c.subject_id and a.icustay_id=c.icustay_id
 where b.itemid in (220180,220179,
                    646, 220210, 220045, 198)
and a.hadm_id is not null)
select * from labs
union all
select * from chartsevents); 


select * from mimiciii.readmission where subject_id not in (select distinct subject_id from mimiciii.timeseries_table);


-- Avg events within an hour
--4.--Avg measurements done in the same hour and remove dupes all at once ################ 

DROP MATERIALIZED VIEW IF EXISTS mimiciii.timseries_table_avg CASCADE;
CREATE MATERIALIZED VIEW mimiciii.timseries_table_avg as(
with hours_entered as(
select a.*,round(cast((EXTRACT(DAY from (a.charttime - b.min_time)) * 24 +
                        EXTRACT(HOUR from (a.charttime - b.min_time))+
                        EXTRACT(MINUTE from (a.charttime - b.min_time))/60) as numeric),0) as hours_in,
       (EXTRACT(DAY from (a.charttime - b.min_time)) * 24 +
                        EXTRACT(HOUR from (a.charttime - b.min_time))+
                        EXTRACT(MINUTE from (a.charttime - b.min_time))/60) as hours_raw
from mimiciii.timeseries_table a
left join 
		(select subject_id,label,
		       min(charttime) as min_time 
		from mimiciii.timeseries_table
		group by subject_id, label) b
		on a.subject_id=b.subject_id and a.label=b.label)
select b.*
from (select *, 
             round(cast(AVG(valuenum) over (partition by subject_id, label, hours_in order by itemid) as decimal), 2) as valuenum_avg,
             row_number() over (partition by subject_id, hours_in, "label") as repeated_measures
      from hours_entered) as b
where b.repeated_measures=1);

