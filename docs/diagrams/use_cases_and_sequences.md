# Hành vi & Tương tác Hệ thống (Use Cases, Sequence & State)

Tài liệu này chứa các sơ đồ mô tả hành vi, các tác vụ người dùng và vòng đời thực thể của **Churn Prediction Platform**.

---

## 1. Diagram 2: UseCase Diagram (Sơ đồ Tác vụ Người dùng)

### Mô tả
Use Case Diagram thể hiện các chức năng của hệ thống được phân rã cho 3 Actor chính:
*   **Data Analyst**: Upload dataset, Profile dataset, Xác nhận CPI Target, Re-evaluate leakage, Kích hoạt train model, Promote model, Kiểm tra data drift.
*   **Business User**: Thực hiện dự đoán realtime (Single prediction), dự đoán lô (Batch prediction), xem mức độ quan trọng tính năng (Feature importance), phân nhóm A/B.
*   **Ops Engineer**: Kiểm tra sức khỏe hệ thống (Health Check), thu thập metrics, xem báo cáo tóm tắt các job.

### Preview
![UseCase Diagram](../img/UseCase.drawio.png)

### draw.io XML
```xml
<mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1654" pageHeight="1169" math="0" shadow="0">
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>

    <!-- Title -->
    <mxCell id="t0" value="Churn Prediction Platform — Use Case Diagram" style="text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="300" y="20" width="900" height="40" as="geometry"/>
    </mxCell>

    <!-- System Boundary -->
    <mxCell id="sys" value="Churn Prediction Platform API" style="swimlane;startSize=30;fillColor=#f5f5f5;strokeColor=#666666;fontSize=14;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="200" y="80" width="900" height="880" as="geometry"/>
    </mxCell>

    <!-- Actors -->
    <mxCell id="actor_da" value="Data Analyst" style="shape=mxgraph.flowchart.actor;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;" vertex="1" parent="1">
      <mxGeometry x="60" y="250" width="60" height="80" as="geometry"/>
    </mxCell>
    <mxCell id="actor_bu" value="Business User" style="shape=mxgraph.flowchart.actor;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=12;" vertex="1" parent="1">
      <mxGeometry x="60" y="600" width="60" height="80" as="geometry"/>
    </mxCell>
    <mxCell id="actor_ops" value="Ops Engineer" style="shape=mxgraph.flowchart.actor;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=12;" vertex="1" parent="1">
      <mxGeometry x="1180" y="400" width="60" height="80" as="geometry"/>
    </mxCell>

    <!-- Data Analyst Use Cases -->
    <mxCell id="uc1" value="Upload Dataset" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="sys">
      <mxGeometry x="80" y="50" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc2" value="Profile Dataset" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="sys">
      <mxGeometry x="80" y="120" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc3" value="Confirm Composite Target (CPI)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="sys">
      <mxGeometry x="80" y="190" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc4" value="Select Excel Sheet" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="sys">
      <mxGeometry x="80" y="260" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc5" value="Re-evaluate Leakage" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="sys">
      <mxGeometry x="80" y="330" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc6" value="Start Training" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="sys">
      <mxGeometry x="80" y="400" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc7" value="Check Training Status" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="sys">
      <mxGeometry x="80" y="470" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc8" value="Manage Model Versions&#xa;(List / Promote / Compare)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="sys">
      <mxGeometry x="80" y="540" width="180" height="60" as="geometry"/>
    </mxCell>
    <mxCell id="uc9" value="Check Data Drift" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="sys">
      <mxGeometry x="80" y="620" width="180" height="50" as="geometry"/>
    </mxCell>

    <!-- Business User Use Cases -->
    <mxCell id="uc10" value="Single Prediction" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="sys">
      <mxGeometry x="360" y="540" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc11" value="Batch Prediction" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="sys">
      <mxGeometry x="360" y="610" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc12" value="View Feature Importance" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="sys">
      <mxGeometry x="360" y="680" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc13" value="A/B Group Assignment" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="sys">
      <mxGeometry x="360" y="750" width="180" height="50" as="geometry"/>
    </mxCell>

    <!-- Ops Use Cases -->
    <mxCell id="uc14" value="System Health Check" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="sys">
      <mxGeometry x="630" y="200" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc15" value="View System Metrics" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="sys">
      <mxGeometry x="630" y="270" width="180" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="uc16" value="Job Summary Report" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="sys">
      <mxGeometry x="630" y="340" width="180" height="50" as="geometry"/>
    </mxCell>

    <!-- Associations -->
    <mxCell id="ae1" style="edgeStyle=none;" edge="1" source="actor_da" target="uc1" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="ae2" style="edgeStyle=none;" edge="1" source="actor_da" target="uc2" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="ae3" style="edgeStyle=none;" edge="1" source="actor_da" target="uc6" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="ae4" style="edgeStyle=none;" edge="1" source="actor_da" target="uc8" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="ae5" style="edgeStyle=none;" edge="1" source="actor_bu" target="uc10" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="ae6" style="edgeStyle=none;" edge="1" source="actor_bu" target="uc11" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="ae7" style="edgeStyle=none;" edge="1" source="actor_bu" target="uc13" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="ae8" style="edgeStyle=none;" edge="1" source="actor_ops" target="uc14" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="ae9" style="edgeStyle=none;" edge="1" source="actor_ops" target="uc15" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="ae10" style="edgeStyle=none;" edge="1" source="actor_ops" target="uc16" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
  </root>
</mxGraphModel>
```

---

## 2. Diagram 3: Sequence Diagrams (5 Luồng Tương tác chính)

### Preview
![Sequence Diagrams](../img/Sequence.drawio.png)

Tải mã nguồn XML bên dưới cho từng luồng cụ thể:

### Luồng 1: Upload → Profile
Xem XML trong artifact `diagrams_fixed` — Luồng 1 (Upload → Profile, swim lane layout).

### Luồng 2: Train → AutoML (Background)
Xem XML trong artifact `diagrams_fixed` — Luồng 2 (background task).

### Luồng 3: Batch Predict
Xem XML trong artifact `diagrams_fixed` — Luồng 3 (cache double-check).

### Luồng 4: Drift Check Loop
Xem XML trong artifact `diagrams_fixed` — Luồng 4 (auto-retrain).

### Luồng 5: A/B Testing
Xem XML trong artifact `diagrams_fixed` — Luồng 5 (phân nhóm deterministic).

---

## 3. Diagram M2: State Machine Lifecycle (Vòng đời Trạng thái Thực thể)

### Mô tả
Sơ đồ này mô tả 3 máy trạng thái song song:
1.  **Dataset Lifecycle**: Từ trạng thái `uploaded` → `profiled` (xử lý qua sheet selection nếu cần) → `training` → kết thúc ở `completed` hoặc `failed`.
2.  **TrainingJob Lifecycle**: Khi nhận lệnh, job ở trạng thái `training` (chưa active), chuyển sang `completed` (kèm model_uri, roc_auc) hoặc `failed` (ghi nhận error_message). Khi được promote, thuộc tính `is_active` đổi sang `True` (và các job cũ hơn của dataset sẽ tự động đổi về `False`).
3.  **Drift Loop & Auto-Retrain Trigger**: Mô tả vòng lặp kiểm tra drift của background task. Khi drift vượt ngưỡng (`PSI >= 0.2` hoặc `KS p < 0.05`), job train mới tự động được sinh ra.

### Preview
![State Machine](../img/Stage%20Machine(Entity%20Lifecircure).drawio.png)

### draw.io XML
```xml
<mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1654" pageHeight="827" math="0" shadow="0">
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>

    <mxCell id="2" value="M2 — State Machine: Entity Lifecycle" style="text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=20;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="200" y="10" width="1100" height="36" as="geometry"/>
    </mxCell>

    <!-- ══ DATASET STATE MACHINE ══ -->
    <mxCell id="10" value="Dataset Lifecycle" style="swimlane;startSize=34;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=14;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="20" y="60" width="460" height="700" as="geometry"/>
    </mxCell>

    <!-- Initial state -->
    <mxCell id="11" value="" style="ellipse;aspect=fixed;fillColor=#000000;strokeColor=#000000;" vertex="1" parent="10">
      <mxGeometry x="210" y="40" width="24" height="24" as="geometry"/>
    </mxCell>

    <mxCell id="12" value="UPLOADED" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=13;fontStyle=1;" vertex="1" parent="10">
      <mxGeometry x="155" y="100" width="150" height="54" as="geometry"/>
    </mxCell>

    <!-- Excel multi-sheet branch -->
    <mxCell id="13" value="requires_sheet_selection" style="rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;" vertex="1" parent="10">
      <mxGeometry x="155" y="185" width="150" height="60" as="geometry"/>
    </mxCell>
    <mxCell id="14" value="SELECT-SHEET" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;" vertex="1" parent="10">
      <mxGeometry x="20" y="185" width="120" height="50" as="geometry"/>
    </mxCell>

    <mxCell id="15" value="PROFILED" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=13;fontStyle=1;" vertex="1" parent="10">
      <mxGeometry x="155" y="290" width="150" height="54" as="geometry"/>
    </mxCell>

    <mxCell id="16" value="TRAINING" style="ellipse;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=13;fontStyle=1;" vertex="1" parent="10">
      <mxGeometry x="155" y="390" width="150" height="54" as="geometry"/>
    </mxCell>

    <mxCell id="17" value="COMPLETED" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=13;fontStyle=1;strokeWidth=3;" vertex="1" parent="10">
      <mxGeometry x="155" y="490" width="150" height="54" as="geometry"/>
    </mxCell>

    <mxCell id="18" value="FAILED" style="ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=13;fontStyle=1;strokeWidth=3;" vertex="1" parent="10">
      <mxGeometry x="310" y="490" width="130" height="54" as="geometry"/>
    </mxCell>

    <!-- Dataset transitions -->
    <mxCell id="150" value="POST /upload" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#6c8ebf;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="11" target="12" parent="10"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="151" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#6c8ebf;fontSize=10;" edge="1" source="12" target="13" parent="10"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="152" value="YES" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#d6b656;fontSize=10;" edge="1" source="13" target="14" parent="10"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="153" value="POST /select-sheet" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#d6b656;fontSize=10;" edge="1" source="14" target="15" parent="10"><mxGeometry relative="1" as="geometry"><Array as="points"><mxPoint x="80" y="370"/><mxPoint x="230" y="370"/></Array></mxGeometry></mxCell>
    <mxCell id="154" value="NO / POST /profile" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#82b366;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="13" target="15" parent="10"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="155" value="POST /train" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#d79b00;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="15" target="16" parent="10"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="156" value="training success" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#82b366;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="16" target="17" parent="10"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="157" value="training error&#xa;or schema fail" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#b85450;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="16" target="18" parent="10"><mxGeometry relative="1" as="geometry"/></mxCell>

    <!-- Internal state notes -->
    <mxCell id="19" value="Entry: sanitize_filename(), parse_file()&#xa;R2 path saved to DB" style="text;html=1;strokeColor=#6c8ebf;fillColor=#eff7ff;fontSize=9;align=left;rounded=1;" vertex="1" parent="10">
      <mxGeometry x="20" y="580" width="420" height="36" as="geometry"/>
    </mxCell>
    <mxCell id="19b" value="PROFILED Entry: run_profiling() 3-layer, ColumnProfile[] saved" style="text;html=1;strokeColor=#82b366;fillColor=#eaf5ea;fontSize=9;align=left;rounded=1;" vertex="1" parent="10">
      <mxGeometry x="20" y="622" width="420" height="28" as="geometry"/>
    </mxCell>
    <mxCell id="19c" value="TRAINING Entry: background_tasks.add_task(run_training_sync, timeout=30min)" style="text;html=1;strokeColor=#d79b00;fillColor=#fff8ec;fontSize=9;align=left;rounded=1;" vertex="1" parent="10">
      <mxGeometry x="20" y="656" width="420" height="28" as="geometry"/>
    </mxCell>

    <!-- ══ TRAINING JOB STATE MACHINE ══ -->
    <mxCell id="20" value="TrainingJob Lifecycle" style="swimlane;startSize=34;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=14;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="500" y="60" width="430" height="700" as="geometry"/>
    </mxCell>

    <mxCell id="21" value="" style="ellipse;aspect=fixed;fillColor=#000000;strokeColor=#000000;" vertex="1" parent="20">
      <mxGeometry x="195" y="40" width="24" height="24" as="geometry"/>
    </mxCell>

    <mxCell id="22" value="TRAINING&#xa;(is_active=False)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=12;fontStyle=1;" vertex="1" parent="20">
      <mxGeometry x="140" y="100" width="170" height="64" as="geometry"/>
    </mxCell>

    <mxCell id="23" value="COMPLETED&#xa;(is_active=False)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=12;fontStyle=1;strokeWidth=3;" vertex="1" parent="20">
      <mxGeometry x="60" y="240" width="170" height="64" as="geometry"/>
    </mxCell>

    <mxCell id="24" value="FAILED&#xa;(error_message set)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=12;fontStyle=1;strokeWidth=3;" vertex="1" parent="20">
      <mxGeometry x="250" y="240" width="160" height="64" as="geometry"/>
    </mxCell>

    <!-- is_active transitions -->
    <mxCell id="25" value="COMPLETED&#xa;(is_active=TRUE)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=12;fontStyle=1;strokeWidth=4;" vertex="1" parent="20">
      <mxGeometry x="100" y="380" width="200" height="64" as="geometry"/>
    </mxCell>

    <mxCell id="26" value="COMPLETED&#xa;(is_active=FALSE)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=12;fontStyle=2;" vertex="1" parent="20">
      <mxGeometry x="100" y="500" width="200" height="64" as="geometry"/>
    </mxCell>

    <!-- TrainingJob transitions -->
    <mxCell id="160" value="POST /jobs/datasets/{id}/train" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#d79b00;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="21" target="22" parent="20"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="161" value="run_automl() success&#xa;model_uri, roc_auc saved" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#82b366;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="22" target="23" parent="20"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="162" value="Exception raised&#xa;(timeout/schema/training)" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#b85450;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="22" target="24" parent="20"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="163" value="POST /models/{id}/promote" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#82b366;strokeWidth=3;fontStyle=1;fontSize=10;" edge="1" source="23" target="25" parent="20"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="164" value="new job promoted&#xa;for same dataset" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#666666;strokeWidth=1;dashed=1;fontSize=10;" edge="1" source="25" target="26" parent="20"><mxGeometry relative="1" as="geometry"/></mxCell>

    <!-- Notes -->
    <mxCell id="27" value="⚠️ Idempotency: nếu job status='training' đang tồn tại → return existing job (không tạo mới)" style="text;html=1;strokeColor=#d79b00;fillColor=#fff8ec;fontSize=9;align=left;rounded=1;" vertex="1" parent="20">
      <mxGeometry x="15" y="590" width="400" height="36" as="geometry"/>
    </mxCell>
    <mxCell id="28" value="promote: set is_active=True cho job này, set is_active=False cho tất cả jobs khác của cùng dataset_id" style="text;html=1;strokeColor=#82b366;fillColor=#eaf5ea;fontSize=9;align=left;rounded=1;" vertex="1" parent="20">
      <mxGeometry x="15" y="632" width="400" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="29" value="timeout: MAX_TRAINING_MINUTES=30 → job.status='failed', error_message='Training timeout'" style="text;html=1;strokeColor=#b85450;fillColor=#fef0ee;fontSize=9;align=left;rounded=1;" vertex="1" parent="20">
      <mxGeometry x="15" y="678" width="400" height="28" as="geometry"/>
    </mxCell>

    <!-- ══ DRIFT AUTO-RETRAIN LOOP ══ -->
    <mxCell id="30" value="Auto-Retrain Trigger" style="swimlane;startSize=34;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=14;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="950" y="60" width="340" height="700" as="geometry"/>
    </mxCell>

    <mxCell id="31" value="App Running&#xa;DRIFT_AUTO_RETRAIN=true&#xa;ENABLE_DRIFT_SCHEDULER=true" style="ellipse;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=11;fontStyle=1;" vertex="1" parent="30">
      <mxGeometry x="60" y="50" width="220" height="70" as="geometry"/>
    </mxCell>

    <mxCell id="32" value="Drift Loop Running&#xa;(asyncio background task)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;" vertex="1" parent="30">
      <mxGeometry x="70" y="160" width="200" height="60" as="geometry"/>
    </mxCell>

    <mxCell id="33" value="sleep(DRIFT_CHECK_INTERVAL&#xa;= 3600s)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=11;" vertex="1" parent="30">
      <mxGeometry x="70" y="270" width="200" height="60" as="geometry"/>
    </mxCell>

    <mxCell id="34" value="PSI ≥ 0.2 OR&#xa;KS p &lt; 0.05?" style="rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;fontStyle=1;" vertex="1" parent="30">
      <mxGeometry x="80" y="370" width="180" height="80" as="geometry"/>
    </mxCell>

    <mxCell id="35" value="Auto-Retrain Triggered&#xa;POST /jobs/.../train" style="ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;fontStyle=1;strokeWidth=2;" vertex="1" parent="30">
      <mxGeometry x="70" y="490" width="200" height="60" as="geometry"/>
    </mxCell>

    <mxCell id="36" value="Cleanup Inference Files&#xa;&gt; INFERENCE_RETENTION_DAYS=30" style="ellipse;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=10;" vertex="1" parent="30">
      <mxGeometry x="70" y="590" width="200" height="60" as="geometry"/>
    </mxCell>

    <mxCell id="170" value="asyncio.create_task()" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#d79b00;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="31" target="32" parent="30"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="171" value="sleep(10s) startup" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#666666;fontSize=10;" edge="1" source="32" target="33" parent="30"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="172" value="check each dataset" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#d6b656;fontSize=10;" edge="1" source="33" target="34" parent="30"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="173" value="YES drift" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#b85450;strokeWidth=2;fontStyle=1;fontSize=10;" edge="1" source="34" target="35" parent="30"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="174" value="after check" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#666666;fontSize=10;" edge="1" source="34" target="36" parent="30"><mxGeometry relative="1" as="geometry"><Array as="points"><mxPoint x="290" y="410"/><mxPoint x="290" y="620"/></Array></mxGeometry></mxCell>
    <mxCell id="175" value="sleep(3600s)" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;strokeColor=#666666;strokeWidth=2;fontStyle=1;fontSize=10;dashed=1;" edge="1" source="36" target="33" parent="30"><mxGeometry relative="1" as="geometry"><Array as="points"><mxPoint x="20" y="620"/><mxPoint x="20" y="300"/></Array></mxGeometry></mxCell>

  </root>
</mxGraphModel>
```
