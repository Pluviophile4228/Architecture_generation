import pandas as pd
import json
import os
import re


class FHAPreprocessor:
    def __init__(self, arch_json_path):
        self.cluster_centers = self._load_cluster_centers(arch_json_path)
        print(f"[INFO] Loaded functional cluster centers: {self.cluster_centers}")

        # Mapping rules: Ensure Values match 'implemented_functions' in initial_arch.json exactly
        # This allows Arch.py to automatically allocate requirements to components

        self.mapping_rules = {
            # --- 1. Execution Function Domain -> Pitch Control Actuation ---
            "Motor": "Pitch Control Actuation",
            "Servo": "Pitch Control Actuation",
            "Actuation": "Pitch Control Actuation",
            "Elevator": "Pitch Control Actuation",
            "Surface": "Pitch Control Actuation",

            # --- 2. Sensing Function Domain -> Air Data Acquisition ---
            "Air Data": "Air Data Acquisition",
            "Altitude": "Air Data Acquisition",
            "Speed": "Air Data Acquisition",
            "AOA": "Air Data Acquisition",
            "Angle": "Air Data Acquisition",
            "Sensor": "Air Data Acquisition",

            # --- 3. Logic Function Domain -> Flight Control Laws ---
            "Flight Control": "Flight Control Laws",
            "Control Law": "Flight Control Laws",
            "Trajectory": "Flight Control Laws",
            "Attitude": "Flight Control Laws",
            "Roll": "Flight Control Laws",
            "Yaw": "Flight Control Laws",
            "Pitch": "Flight Control Laws",  # Note: 'Pitch' can be Attitude (FCC) or Surface (Actuation) depending on context priority
            "Thrust": "Flight Control Laws",
            "Envelope": "Flight Control Laws",
            "Load": "Flight Control Laws",
            "Stability": "Flight Control Laws",
            "Integration": "Flight Control Laws"
        }


    def _load_cluster_centers(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        clusters = set()
        for comp in data.get("components", []):
            for func in comp.get("implemented_functions", []):
                clusters.add(func)
        return list(clusters)

    def execute_clustering(self, fha_excel_path, output_path):
        print(f"\n[INFO] Reading FHA table: {fha_excel_path} ...")
        df = pd.read_excel(fha_excel_path)

        # Fill NaN values caused by merged cells
        df['Top Function'] = df['Top Function'].ffill()
        df['Sub-Function'] = df['Sub-Function'].ffill()

        def match_cluster(row):
            text = str(row['Sub-Function']) + " " + str(row['Top Function'])
            for kw, cluster in self.mapping_rules.items():
                if kw.lower() in text.lower():
                    return cluster
            return "Unassigned"

        df['Mapped_Cluster'] = df.apply(match_cluster, axis=1)

        print("Clustering Statistics:")
        print(df['Mapped_Cluster'].value_counts())

        df.to_excel(output_path, index=False)
        print(f"[INFO] Processing complete! Result saved to: {output_path}")
        return df


class EnvelopeExtractor:
    def __init__(self):
        # Severity Ranks
        self.severity_rank = {
            "Catastrophic": 4, "Catastrophic (A)": 4,
            "Hazardous": 3, "Hazardous (B)": 3,
            "Major": 2, "Major (C)": 2,
            "Minor": 1, "Minor (D)": 1
        }
        self.rank_to_severity = {4: "Catastrophic", 3: "Hazardous", 2: "Major", 1: "Minor"}

        # Type Priority: DesignError > Integrity > Availability
        self.type_priority = {
            "DesignError": 3,
            "Integrity": 2,
            "Availability": 1
        }

    def _parse_row_attributes(self, row, cluster_name):
        # 1. Parse Severity (L)
        sev_str = str(row['Class']).strip()
        l_val = 1
        if 'A)' in sev_str or 'Catastrophic' in sev_str:
            l_val = 4
        elif 'B)' in sev_str or 'Hazardous' in sev_str:
            l_val = 3
        elif 'C)' in sev_str or 'Major' in sev_str:
            l_val = 2

        # 2. Parse Type (T)
        desc = str(row['Detailed Failure Condition'])
        desc_lower = desc.lower()

        # Determine DesignError: FCC and Catastrophic
        is_design_error = False
        if "Flight Control Laws" in cluster_name and l_val == 4:
            is_design_error = True

        # Determine Basic Type
        # Keyword list to ensure "Loss" is not misclassified as Integrity
        integrity_keywords = ["uncommanded", "erroneous", "hardover", "oscillation", "stuck", "false", "misleading"]

        if any(x in desc_lower for x in integrity_keywords):
            t_type = "Integrity"
        else:
            # Default to Availability (includes loss, fail, exceedance if no specific integrity keywords exist)
            t_type = "Availability"

        if is_design_error:
            t_type = "DesignError"

        return l_val, t_type, desc

    def process_groups(self, df):
        print("\n[INFO] Executing Envelope Extraction (Worst-Case) ...")
        envelopes = []

        groups = df[df['Mapped_Cluster'] != 'Unassigned'].groupby('Mapped_Cluster')

        for cluster_name, group_df in groups:
            # --- Core Logic Start ---
            # Instead of separate max_l and max_t, find the "Worst-Case Row"
            best_row_l = -1
            best_row_t_val = -1

            final_severity = "Minor"
            final_focus = "Availability"
            final_design_error = False
            final_desc = cluster_name

            for _, row in group_df.iterrows():
                l_val, t_str, desc = self._parse_row_attributes(row, cluster_name)
                t_val = self.type_priority.get(t_str, 1)

                # Logic: Prioritize Severity (L), then Type Priority (T)
                update = False
                if l_val > best_row_l:
                    update = True
                elif l_val == best_row_l:
                    # If Severity is equal, take higher Type Priority (Design > Integrity > Availability)
                    if t_val > best_row_t_val:
                        update = True

                if update:
                    best_row_l = l_val
                    best_row_t_val = t_val

                    final_severity = self.rank_to_severity.get(l_val, "Minor")
                    final_desc = desc

                    # Map Focus
                    if t_str == "DesignError":
                        final_design_error = True
                        final_focus = "Integrity"
                    elif t_str == "Integrity":
                        final_design_error = False
                        final_focus = "Integrity"
                    else:
                        final_design_error = False
                        final_focus = "Availability"
            # --- Core Logic End ---

            # Construct Envelope Vector
            envelope = {
                "related_function": cluster_name,
                "severity": final_severity,
                "focus": final_focus,
                "design_error": final_design_error,
                "description": final_desc,
                "debug_info": f"<{final_severity}, {final_focus}>"
            }
            envelopes.append(envelope)

            print(f"  [Cluster: {cluster_name}] -> Envelope: {envelope['debug_info']}")

        return envelopes

    def save_to_json(self, envelopes, output_path):
        json_data = []
        for i, env in enumerate(envelopes):
            json_data.append({
                "id": f"REQ-{i + 1:03d}",
                "related_function": env['related_function'],
                "description": env['description'],
                "severity": env['severity'],
                "focus": env['focus'],
                "design_error": env['design_error']
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"\n[INFO] Solver input file generated: {output_path}")

# --- Main Entry Point ---
if __name__ == "__main__":
    arch_file = "initial_arch.json"
    fha_input = "Raw_FHA_Input.xlsx"
    fha_output = "Processed_FHA_Clustered.xlsx"
    solver_input_json = "safety_requirements.json"

    if os.path.exists(arch_file) and os.path.exists(fha_input):
        # 1. Clustering
        preprocessor = FHAPreprocessor(arch_file)
        clustered_df = preprocessor.execute_clustering(fha_input, fha_output)

        # 2. Extract Envelopes
        extractor = EnvelopeExtractor()
        envelopes = extractor.process_groups(clustered_df)

        # 3. Save Results
        extractor.save_to_json(envelopes, solver_input_json)

    else:
        print("[ERROR] Input files not found. Ensure initial_arch.json and Raw_FHA_Input.xlsx exist.")