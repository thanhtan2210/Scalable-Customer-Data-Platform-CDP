# Cấu trúc Dữ liệu & Đánh giá (ER, MTL & Evaluation Framework)

Tài liệu này chứa các sơ đồ liên quan đến cấu trúc cơ sở dữ liệu, mạng neural Multi-Task Learning và khung đánh giá kiểm thử mô hình của **Churn Prediction Platform**.

---

## 1. Diagram 5: Entity-Relationship (ER) Diagram

### Mô tả
Sơ đồ cơ sở dữ liệu thực tế mô tả cấu trúc của 4 bảng trong PostgreSQL và 5 phiên bản migrations của Alembic:
*   `datasets`: Lưu thông tin file dữ liệu thô tải lên (`r2_path`, `filename`, `row_count`, `col_count`, `status`). Có index trên `user_id`.
*   `profiles`: Quan hệ 1-1 với `datasets`. Lưu trữ JSON danh sách `ColumnProfile` của dataset và đề xuất target column.
*   `training_jobs`: Lưu các lượt training của dataset. Ghi nhận trạng thái (`status: training/completed/failed`), model URI (`model_uri` từ MLflow), độ chính xác (`roc_auc`), threshold tối ưu, thời gian chạy, cờ active (`is_active` để serving định tuyến) và các tags metadata.
*   `drift_reports`: Ghi nhận lịch sử kiểm tra data drift của từng dataset, lưu trữ JSON metrics tính toán (PSI, p-value) và cờ `drift_detected`.

### Preview
![ER Diagram](../img/ER%20diagram.drawio.png)

### draw.io XML
```xml
<mxGraphModel dx="1422" dy="762" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
  <root>
    <mxCell id="0"/><mxCell id="1" parent="0"/>
    <mxCell id="t0" value="ER Diagram — Churn Prediction Platform DB" style="text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="200" y="10" width="700" height="30" as="geometry"/>
    </mxCell>

    <!-- datasets table -->
    <mxCell id="tbl_ds" value="datasets" style="shape=table;startSize=30;container=1;collapsible=1;childLayout=tableLayout;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=13;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="80" y="80" width="260" height="230" as="geometry"/>
    </mxCell>
    <mxCell id="r_ds1" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;fontStyle=0;spacing=2;" vertex="1" parent="tbl_ds">
      <mxGeometry y="30" width="260" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="ds_id" value="🔑 id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;fontStyle=1;overflow=hidden;" vertex="1" connectable="0" parent="r_ds1"><mxGeometry width="130" height="26" as="geometry"/></mxCell>
    <mxCell id="ds_id_t" value="String (UUID) PK" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds1"><mxGeometry x="130" width="130" height="26" as="geometry"/></mxCell>

    <mxCell id="r_ds2" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_ds"><mxGeometry y="56" width="260" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="ds_uid" value="user_id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds2"><mxGeometry width="130" height="26" as="geometry"/></mxCell>
    <mxCell id="ds_uid_t" value="String, INDEX" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds2"><mxGeometry x="130" width="130" height="26" as="geometry"/></mxCell>

    <mxCell id="r_ds3" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_ds"><mxGeometry y="82" width="260" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="ds_fn" value="filename" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds3"><mxGeometry width="130" height="26" as="geometry"/></mxCell>
    <mxCell id="ds_fn_t" value="String NOT NULL" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds3"><mxGeometry x="130" width="130" height="26" as="geometry"/></mxCell>

    <mxCell id="r_ds4" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_ds"><mxGeometry y="108" width="260" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="ds_rp" value="r2_path" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds4"><mxGeometry width="130" height="26" as="geometry"/></mxCell>
    <mxCell id="ds_rp_t" value="String NOT NULL" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds4"><mxGeometry x="130" width="130" height="26" as="geometry"/></mxCell>

    <mxCell id="r_ds5" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_ds"><mxGeometry y="134" width="260" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="ds_rc" value="row_count" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds5"><mxGeometry width="130" height="26" as="geometry"/></mxCell>
    <mxCell id="ds_rc_t" value="Integer nullable" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds5"><mxGeometry x="130" width="130" height="26" as="geometry"/></mxCell>

    <mxCell id="r_ds6" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_ds"><mxGeometry y="160" width="260" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="ds_st" value="status" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds6"><mxGeometry width="130" height="26" as="geometry"/></mxCell>
    <mxCell id="ds_st_t" value="String def='uploaded'" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds6"><mxGeometry x="130" width="130" height="26" as="geometry"/></mxCell>

    <mxCell id="r_ds7" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_ds"><mxGeometry y="186" width="260" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="ds_ca" value="created_at" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds7"><mxGeometry width="130" height="26" as="geometry"/></mxCell>
    <mxCell id="ds_ca_t" value="DateTime def=utcnow" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="r_ds7"><mxGeometry x="130" width="130" height="26" as="geometry"/></mxCell>

    <!-- profiles table -->
    <mxCell id="tbl_pr" value="profiles" style="shape=table;startSize=30;container=1;collapsible=1;childLayout=tableLayout;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=13;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="420" y="80" width="280" height="160" as="geometry"/>
    </mxCell>
    <mxCell id="pr_r1" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_pr"><mxGeometry y="30" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_id" value="🔑 id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;fontStyle=1;overflow=hidden;" vertex="1" connectable="0" parent="pr_r1"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_id_t" value="String (UUID) PK" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="pr_r1"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_r2" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_pr"><mxGeometry y="56" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_did" value="🔗 dataset_id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="pr_r2"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_did_t" value="FK→datasets.id UNIQUE" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="pr_r2"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_r3" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_pr"><mxGeometry y="82" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_pj" value="profiles_json" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="pr_r3"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_pj_t" value="JSON (List[ColumnProfile])" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="pr_r3"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_r4" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_pr"><mxGeometry y="108" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_st" value="suggested_target" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="pr_r4"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_st_t" value="String nullable" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="pr_r4"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_r5" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_pr"><mxGeometry y="134" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_ca" value="created_at" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="pr_r5"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="pr_ca_t" value="DateTime def=utcnow" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="pr_r5"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>

    <!-- training_jobs table -->
    <mxCell id="tbl_tj" value="training_jobs" style="shape=table;startSize=30;container=1;collapsible=1;childLayout=tableLayout;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=13;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="80" y="360" width="280" height="420" as="geometry"/>
    </mxCell>
    <mxCell id="tj_r1" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="30" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_id" value="🔑 id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;fontStyle=1;overflow=hidden;" vertex="1" connectable="0" parent="tj_r1"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_id_t" value="String (UUID) PK" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r1"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r2" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="56" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_did" value="🔗 dataset_id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r2"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_did_t" value="FK→datasets.id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r2"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r3" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="82" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_st" value="status" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r3"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_st_t" value="String def='training'" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r3"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r4" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="108" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_mu" value="model_uri" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r4"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_mu_t" value="String nullable" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r4"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r5" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="134" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_ra" value="roc_auc" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r5"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_ra_t" value="Float nullable" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r5"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r6" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="160" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_ot" value="optimal_threshold" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r6"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_ot_t" value="Float nullable" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r6"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r7" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="186" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_tc" value="target_column" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r7"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_tc_t" value="String nullable" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r7"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r8" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="212" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_sa" value="started_at" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r8"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_sa_t" value="DateTime def=utcnow" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r8"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r9" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="238" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_fa" value="finished_at" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r9"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_fa_t" value="DateTime nullable" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r9"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r10" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="264" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_ia" value="is_active" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r10"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_ia_t" value="Boolean def=False" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r10"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_r11" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_tj"><mxGeometry y="290" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_tg" value="tags" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r11"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="tj_tg_t" value="JSON def={}" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="tj_r11"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>

    <!-- drift_reports table -->
    <mxCell id="tbl_dr" value="drift_reports" style="shape=table;startSize=30;container=1;collapsible=1;childLayout=tableLayout;fillColor=#f8cecc;strokeColor=#b85450;fontSize=13;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="420" y="360" width="280" height="240" as="geometry"/>
    </mxCell>
    <mxCell id="dr_r1" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_dr"><mxGeometry y="30" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_id" value="🔑 id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;fontStyle=1;overflow=hidden;" vertex="1" connectable="0" parent="dr_r1"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_id_t" value="String (UUID) PK" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r1"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_r2" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_dr"><mxGeometry y="56" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_did" value="🔗 dataset_id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r2"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_did_t" value="FK→datasets.id" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r2"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_r3" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_dr"><mxGeometry y="82" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_rr" value="reference_rows" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r3"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_rr_t" value="Integer" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r3"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_r4" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_dr"><mxGeometry y="108" width="280" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_tr" value="target_rows" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r4"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_tr_t" value="Integer" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r4"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_r5" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_dr"><mxGeometry y="134" width="280" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="dr_dd" value="drift_detected" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r5"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_dd_t" value="Boolean" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r5"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_r6" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_dr"><mxGeometry y="160" width="280" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="dr_m" value="metrics" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r6"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_m_t" value="JSON" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r6"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_r7" value="" style="shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;fillColor=none;collapsible=0;dropTarget=0;" vertex="1" parent="tbl_dr"><mxGeometry y="186" width="280" height="26" as="geometry"/>
    </mxCell>
    <mxCell id="dr_ca" value="created_at" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r7"><mxGeometry width="140" height="26" as="geometry"/></mxCell>
    <mxCell id="dr_ca_t" value="DateTime def=utcnow" style="shape=partialRectangle;top=0;left=0;right=0;bottom=0;overflow=hidden;" vertex="1" connectable="0" parent="dr_r7"><mxGeometry x="140" width="140" height="26" as="geometry"/></mxCell>

    <!-- Relationships -->
    <mxCell id="rel1" value="1" style="edgeStyle=entityRelationEdgeStyle;endArrow=ERzeroToMany;startArrow=ERmandOne;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" source="tbl_ds" target="tbl_pr" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="rel2" value="1..N" style="edgeStyle=entityRelationEdgeStyle;endArrow=ERzeroToMany;startArrow=ERmandOne;exitX=0;exitY=1;entryX=0;entryY=0;" edge="1" source="tbl_ds" target="tbl_tj" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="rel3" value="1..N" style="edgeStyle=entityRelationEdgeStyle;endArrow=ERzeroToMany;startArrow=ERmandOne;exitX=1;exitY=1;entryX=0;entryY=0;" edge="1" source="tbl_ds" target="tbl_dr" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
  </root>
</mxGraphModel>
```

---

## 2. Diagram M3: Multi-Task Learning (MTL) Architecture

### Mô tả
Kiến trúc mạng Neural 2 đầu ra (`MTLChurnModel` bằng PyTorch):
*   **Shared Preprocessor**: Transformer các trường text, category, numeric thành ma trận input dense $X$.
*   **Task 1 Head (Churn Binary Classification)**: Phân loại nhị phân khách hàng rời bỏ. Đi qua 3 tầng fully connected (tích hợp Batch Normalization, ReLU, Dropout để chống quá khớp) và hàm Sigmoid ở ngõ ra.
*   **Task 2 Head (CPI Regression)**: Dự đoán mức độ churn liên tục (Composite Performance Index). Đi qua 2 tầng fully connected và hàm kích hoạt ReLU ở ngõ ra để tối ưu hóa trị liên tục.
*   **Combined Loss Function**:
    $$L_{total} = \alpha \cdot L_{BCE}(P(churn), y) + \beta \cdot L_{MSE}(CPI_{pred}, y_{cpi})$$
    Giúp tối ưu hóa biểu diễn đặc trưng (shared representation) phục vụ cả hai tác vụ.

### Preview
![MTL Architecture](../img/MTL%20(Multi-Task%20Learning)%20Architecture.drawio.png)

### draw.io XML
Xem mã XML trong artifact `diagrams_missing_M1_M3` (Diagram M3).

---

## 3. Diagram M7: Evaluation Framework (Khung Kiểm thử Đánh giá)

### Mô tả
Khung đánh giá thống nhất của nền tảng:
1.  **Model Selection Metrics**: Cross-validation 5-fold (Stratified KFold) tính trung bình metric ROC-AUC để chọn model tốt nhất. Phân tách ngưỡng tối ưu (optimal_threshold) tìm điểm F1-max trên đường cong Precision-Recall.
2.  **Model Routing Decision Matrix**: Logic phân loại đặc trưng để định tuyến thuật toán thích ứng (Logistic Regression / Random Forest / XGBoost / MTL).
3.  **A/B Testing Framework**: Phân nhóm khách hàng ngẫu nhiên nhưng đảm bảo tính nhất quán (deterministic qua hash SHA-256 mã khách hàng) và log sự kiện (exposure logging) song song trên DB hoặc JSONL file.
4.  **MLflow Tracking**: Vết lưu trữ metadata đầy đủ (Run tags, Metrics, Artifacts, Model registry và cơ chế dọn dẹp giữ lại 5 runs gần nhất).

### Preview
![Evaluation Framework](../img/Evaluation%20Framework%20Diagram.drawio.png)

### draw.io XML
Xem mã XML trong artifact `diagrams_missing_M4_M7` (Diagram M7).
