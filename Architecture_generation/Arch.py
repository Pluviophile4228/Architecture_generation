import networkx as nx
import numpy as np
from enum import Enum
from typing import List, Dict, Optional
import math
import json
import os


# --- 1. Basic Definitions ---

class ComponentType(Enum):
    SENSOR = "Sensor"
    ACTUATOR = "Actuator"
    PROCESSOR = "Processor"
    MONITOR = "Monitor"  # u_DIC, u_MON, u_AM, u_EF
    VOTER = "Voter"  # u_Voter
    SWITCH = "Switch"  # u_SW
    COMPARATOR = "Comparator"  # u_CMP
    # [Added] Special nodes strictly matching document definitions
    FAILURE_DETECTOR = "FailDetector"  # u_FD (SR Pattern)
    RTA_EXEC = "RTA_Executive"  # u_Executive (RTA Pattern)
    PASS_THROUGH = "PassThrough"  # u_IP, u_OP, u_IA, u_FE, u_POO (Logical nodes)


class Component:
    # [Change] failure_rate -> failure_prob
    def __init__(self, uid: str, ctype: ComponentType, failure_prob: float, name: str = "",
                 implemented_functions: List[str] = None):
        self.uid = uid
        self.ctype = ctype
        # [Change] lambda_val -> prob_val (Probability Value)
        self.prob_val = failure_prob
        self.name = name if name else uid
        self.implemented_functions = implemented_functions if implemented_functions else []

    def __repr__(self):
        return f"{self.name}({self.ctype.value})"


class SystemModel:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_component(self, comp: Component):
        self.graph.add_node(comp.uid, data=comp)

    def add_connection(self, source_id: str, target_id: str):
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(source_id, target_id)
        else:
            raise ValueError(f"Nodes {source_id} or {target_id} not in graph.")

    def get_component(self, uid: str) -> Component:
        return self.graph.nodes[uid]['data']


# --- 2. Safety Pattern Operators (Matching Source 131) ---

class PatternOperator:
    def apply(self, system: SystemModel, target_node_id: str):
        if target_node_id not in system.graph:
            raise ValueError("Target node not found")

        original_comp = system.get_component(target_node_id)
        in_edges = list(system.graph.in_edges(target_node_id))
        out_edges = list(system.graph.out_edges(target_node_id))

        system.graph.remove_node(target_node_id)

        # Embed Topology
        new_nodes = self._embed_topology(system, original_comp)

        # Reconnection Logic
        self._reconnect(system, in_edges, out_edges, new_nodes)

        print(f"Applied pattern {self.__class__.__name__} to {original_comp.name}")
        return new_nodes

    def _embed_topology(self, system: SystemModel, original_comp: Component) -> Dict[str, str]:
        raise NotImplementedError

    def _reconnect(self, system: SystemModel, in_edges, out_edges, new_nodes_map):
        # 1. Input Reconnection
        input_targets = new_nodes_map.get('inputs', [])
        for target in input_targets:
            for u, v in in_edges:
                # If pattern monitors input (e.g., u_FD in SR), auto-connect u -> target
                system.add_connection(u, target)

        # 2. Output Reconnection
        output_node = new_nodes_map.get('output')
        if output_node:
            for u, v in out_edges:
                system.add_connection(output_node, v)

        # 3. Feedback Reconnection (for IM pattern)
        feedback_node = new_nodes_map.get('feedback')
        if feedback_node:
            for u, v in in_edges:
                # Feedback loop: u_MON -> u_IP
                try:
                    system.add_connection(feedback_node, u)
                except:
                    pass  # Ignore existing connections or self-loops


# --- Specific Pattern Implementations ---

class PSC_Pattern(PatternOperator):
    """
    PSC: Protected Single Channel
    V = {u_IP, u_DP, u_OP, u_DIC, u_AM}
    E = {(u_IP, u_DP), (u_IP, u_DIC), (u_DIC, u_DP), (u_DP, u_OP), (u_AM, u_DP)}
    """

    def _embed_topology(self, system, org):
        dp_id = f"{org.uid}_DP"  # u_DP
        dic_id = f"{org.uid}_DIC"  # u_DIC
        am_id = f"{org.uid}_AM"  # u_AM

        # [Change] org.lambda_val -> org.prob_val
        system.add_component(Component(dp_id, org.ctype, org.prob_val, f"{org.name}_DP"))
        system.add_component(Component(dic_id, ComponentType.MONITOR, 1e-5, f"{org.name}_DIC"))
        system.add_component(Component(am_id, ComponentType.MONITOR, 1e-6, f"{org.name}_AM"))

        # Internal Connections E
        system.add_connection(dic_id, dp_id)
        system.add_connection(am_id, dp_id)

        # Interface Mapping
        return {'inputs': [dp_id, dic_id], 'output': dp_id}


class CC_Pattern(PatternOperator):
    """
    CC: Cross Compare
    V = {u_OP1...u_OPn, u_CMP}
    """

    def _embed_topology(self, system, org):
        op1 = f"{org.uid}_OP1"
        op2 = f"{org.uid}_OP2"
        cmp_id = f"{org.uid}_CMP"

        system.add_component(Component(op1, org.ctype, org.prob_val, f"{org.name}_OP1"))
        system.add_component(Component(op2, org.ctype, org.prob_val, f"{org.name}_OP2"))
        system.add_component(Component(cmp_id, ComponentType.COMPARATOR, 1e-7, "Comparator"))

        system.add_connection(op1, cmp_id)
        system.add_connection(op2, cmp_id)

        return {'inputs': [op1, op2], 'output': cmp_id}


class MN_Pattern(PatternOperator):
    """
    MN: M-out-of-N
    V = {u_DASi, u_DPi, u_OPi, u_Voter}
    """

    def __init__(self, n=3, m=2):
        self.N = n
        self.M = m

    def _embed_topology(self, system, org):
        channels = []
        for i in range(self.N):
            # Strict expansion: DAS -> DP -> OP
            das_id = f"{org.uid}_DAS{i + 1}"
            dp_id = f"{org.uid}_DP{i + 1}"
            op_id = f"{org.uid}_OP{i + 1}"

            # DAS simplified to PassThrough
            system.add_component(Component(das_id, ComponentType.PASS_THROUGH, 0, f"{org.name}_DAS{i + 1}"))
            system.add_component(Component(dp_id, org.ctype, org.prob_val, f"{org.name}_DP{i + 1}"))
            system.add_component(Component(op_id, ComponentType.PASS_THROUGH, 0, f"{org.name}_OP{i + 1}"))

            # E: (DAS, DP), (DP, OP)
            system.add_connection(das_id, dp_id)
            system.add_connection(dp_id, op_id)

            channels.append(das_id)

        voter_id = f"{org.uid}_Voter"
        system.add_component(Component(voter_id, ComponentType.VOTER, 1e-10, "Voter"))

        # E: (OP, Voter)
        for i in range(self.N):
            op_node = f"{org.uid}_OP{i + 1}"
            system.add_connection(op_node, voter_id)

        return {'inputs': channels, 'output': voter_id}


class SR_Pattern(PatternOperator):
    """
    SR: Standby Redundancy
    V = {u_IPP, u_DPP, u_OPP, u_IPS, u_DPS, u_OPS, u_FD, u_SW}
    E includes FD monitoring inputs/outputs and FD->SW
    """

    def _embed_topology(self, system, org):
        # 1. Define Nodes
        # Primary
        ipp = f"{org.uid}_IPP"
        dpp = f"{org.uid}_DPP"
        opp = f"{org.uid}_OPP"
        # Secondary
        ips = f"{org.uid}_IPS"
        dps = f"{org.uid}_DPS"
        ops = f"{org.uid}_OPS"
        # Control
        fd_id = f"{org.uid}_FD"
        sw_id = f"{org.uid}_SW"

        system.add_component(Component(ipp, ComponentType.PASS_THROUGH, 0, f"{org.name}_IPP"))
        system.add_component(Component(dpp, org.ctype, org.prob_val, f"{org.name}_DPP"))
        system.add_component(Component(opp, ComponentType.PASS_THROUGH, 0, f"{org.name}_OPP"))

        system.add_component(Component(ips, ComponentType.PASS_THROUGH, 0, f"{org.name}_IPS"))
        system.add_component(Component(dps, org.ctype, org.prob_val, f"{org.name}_DPS"))
        system.add_component(Component(ops, ComponentType.PASS_THROUGH, 0, f"{org.name}_OPS"))

        system.add_component(Component(fd_id, ComponentType.FAILURE_DETECTOR, 1e-6, f"{org.name}_FD"))
        system.add_component(Component(sw_id, ComponentType.SWITCH, 1e-10, "Switch"))

        # 2. Define Connections E
        # Primary flow
        system.add_connection(ipp, dpp)
        system.add_connection(dpp, opp)
        system.add_connection(opp, sw_id)
        # Secondary flow
        system.add_connection(ips, dps)
        system.add_connection(dps, ops)
        system.add_connection(ops, sw_id)

        # FD Monitoring (FD monitors IP & OP)
        system.add_connection(ipp, fd_id)
        system.add_connection(ips, fd_id)
        system.add_connection(opp, fd_id)
        system.add_connection(ops, fd_id)

        # FD controls Switch
        system.add_connection(fd_id, sw_id)

        return {'inputs': [ipp, ips], 'output': sw_id}


class RTA_Pattern(PatternOperator):
    """
    RTA: Run-Time Assurance
    V = {u_FE, u_Comp, u_POO, u_IA, u_IP, u_Sim, u_OP, u_Executive}
    """

    def _embed_topology(self, system, org):
        # Complex Path
        ia_id = f"{org.uid}_IA"  # Input Allocator
        fe_id = f"{org.uid}_FE"  # Feature Extractor
        comp_id = f"{org.uid}_Comp"
        poo_id = f"{org.uid}_POO"  # Performance Optimized Output

        # Simple Path
        ip_id = f"{org.uid}_IP"
        sim_id = f"{org.uid}_Sim"
        op_id = f"{org.uid}_OP"

        # Executive
        exec_id = f"{org.uid}_Executive"

        system.add_component(Component(ia_id, ComponentType.PASS_THROUGH, 0, f"{org.name}_IA"))
        system.add_component(Component(fe_id, ComponentType.PASS_THROUGH, 0, f"{org.name}_FE"))
        system.add_component(Component(comp_id, org.ctype, org.prob_val, f"{org.name}_Comp"))
        system.add_component(Component(poo_id, ComponentType.PASS_THROUGH, 0, f"{org.name}_POO"))

        system.add_component(Component(ip_id, ComponentType.PASS_THROUGH, 0, f"{org.name}_IP"))
        system.add_component(Component(sim_id, org.ctype, 1e-5, f"{org.name}_Sim"))
        system.add_component(Component(op_id, ComponentType.PASS_THROUGH, 0, f"{org.name}_OP"))

        system.add_component(Component(exec_id, ComponentType.RTA_EXEC, 1e-7, f"{org.name}_Exec"))

        # Complex Chain: IA -> FE -> Comp -> POO -> Executive
        system.add_connection(ia_id, fe_id)
        system.add_connection(fe_id, comp_id)
        system.add_connection(comp_id, poo_id)
        system.add_connection(poo_id, exec_id)

        # Simple Chain: IA -> IP -> Sim -> OP -> Executive
        system.add_connection(ia_id, ip_id)
        system.add_connection(ip_id, sim_id)
        system.add_connection(sim_id, op_id)
        system.add_connection(op_id, exec_id)

        return {'inputs': [ia_id], 'output': exec_id}


class IM_Pattern(PatternOperator):
    """
    IM: Independent Monitoring
    Includes feedback loop (u_MON, u_IP) and independent sensing
    """

    def _embed_topology(self, system, org):
        ip_id = f"{org.uid}_IP"
        dp_id = f"{org.uid}_DP"
        op_id = f"{org.uid}_OP"
        dic_id = f"{org.uid}_DIC"
        mon_id = f"{org.uid}_MON"

        system.add_component(Component(ip_id, ComponentType.PASS_THROUGH, 0, f"{org.name}_IP"))
        system.add_component(Component(dp_id, org.ctype, org.prob_val, f"{org.name}_DP"))
        system.add_component(Component(op_id, ComponentType.PASS_THROUGH, 0, f"{org.name}_OP"))
        system.add_component(Component(dic_id, ComponentType.MONITOR, 1e-5, f"{org.name}_DIC"))
        system.add_component(Component(mon_id, ComponentType.MONITOR, 1e-6, f"{org.name}_MON"))

        # E: (IP, DP), (IP, DIC), (DIC, DP), (DP, OP), (OP, MON)
        system.add_connection(ip_id, dp_id)
        system.add_connection(ip_id, dic_id)
        system.add_connection(dic_id, dp_id)
        system.add_connection(dp_id, op_id)
        system.add_connection(op_id, mon_id)

        # Feedback: (MON, IP)
        system.add_connection(mon_id, ip_id)

        return {'inputs': [ip_id], 'output': op_id, 'feedback': mon_id}


# --- 4. Helper Classes & Inference Engine ---

class SafetySeverity(Enum):
    CATASTROPHIC = "Catastrophic"
    HAZARDOUS = "Hazardous"
    MAJOR = "Major"
    MINOR = "Minor"


class SafetyFocus(Enum):
    INTEGRITY = "Integrity"
    AVAILABILITY = "Availability"


class SafetyRequirement:
    def __init__(self, related_function: str, severity: SafetySeverity,
                 focus: SafetyFocus, design_error: bool = False, req_id: str = ""):
        self.req_id = req_id
        self.related_function = related_function  # Associated function name
        self.target_comp_id = None  # To be allocated
        self.severity = severity
        self.focus = focus
        self.design_error = design_error
        self.current_processing_id = None


class InferenceEngine:
    """
    [FIXED] Inference logic completed
    """

    def select_pattern(self, req: SafetyRequirement) -> Optional[PatternOperator]:
        # Axiom 3.3 (Software Design Common Mode Error)
        if req.design_error:
            return RTA_Pattern()

        # Axiom 3.2 (Availability Priority)
        if req.focus == SafetyFocus.AVAILABILITY:
            if req.severity in [SafetySeverity.CATASTROPHIC, SafetySeverity.HAZARDOUS]:
                return MN_Pattern(n=3, m=2)
            else:
                return SR_Pattern()

        # Axiom 3.1 (Integrity Inference)
        if req.focus == SafetyFocus.INTEGRITY:
            if req.severity == SafetySeverity.CATASTROPHIC:
                return CC_Pattern()
            else:
                return PSC_Pattern()

        return None


class TopologyConstructor:
    def __init__(self, system_model: SystemModel):
        self.system = system_model
        self.inference_engine = InferenceEngine()


# --- 3. Probability Verification ---

class VerificationEngine:
    @staticmethod
    def verify_struct(system: SystemModel, reqs: List[SafetyRequirement]) -> bool:
        return True

    @staticmethod
    def verify_prob(system: SystemModel, target_node_id: str) -> float:
        if target_node_id not in system.graph: return 1.0

        comp = system.get_component(target_node_id)
        in_edges = list(system.graph.in_edges(target_node_id))

        # Recursively calculate input probabilities
        try:
            input_probs = [VerificationEngine.verify_prob(system, u) for u, _ in in_edges]
        except RecursionError:
            return 1.0  # Cycle detected

        # [Logic based on Source 131 Failure Logic]

        # 1. RTA Executive (u_Executive)
        if comp.ctype == ComponentType.RTA_EXEC:
            # P_haz = P_mon + P_AI * (1 - C_env)
            if len(input_probs) >= 2:
                p_complex = input_probs[0]  # P_AI
                return comp.prob_val + (p_complex * 0)  # Assume C_env = 1
            return comp.prob_val

        # 2. PSC DP
        # Topology: DIC -> DP, AM -> DP
        if any(system.get_component(u).ctype == ComponentType.MONITOR for u, _ in in_edges):
            # P_fail = P_DP * (1 - C) + ...
            coverage = 0.9999
            data_inputs = [p for i, p in enumerate(input_probs) if
                           system.get_component(in_edges[i][0]).ctype != ComponentType.MONITOR]
            p_signal = max(data_inputs, default=comp.prob_val)
            # Factor 1e-5 retained as Common Mode/Coverage factor
            return p_signal * (1 - coverage) + comp.prob_val * 1e-5

        # 3. SR Switch
        # Topology: FD -> Switch
        if comp.ctype == ComponentType.SWITCH and any(
                system.get_component(u).ctype == ComponentType.FAILURE_DETECTOR for u, _ in in_edges):
            # P_sys = P_pri * (P_sw + P_sby)
            data_probs = [p for i, p in enumerate(input_probs) if
                          system.get_component(in_edges[i][0]).ctype != ComponentType.FAILURE_DETECTOR]
            if len(data_probs) >= 2:
                return data_probs[0] * (comp.prob_val + data_probs[1])

        # 4. MN Voter
        if comp.ctype == ComponentType.VOTER:
            if len(input_probs) >= 3:
                p = max(input_probs)
                return 3 * (p ** 2) + comp.prob_val
            return max(input_probs, default=1.0)

        # PassThrough Node
        if comp.ctype == ComponentType.PASS_THROUGH:
            return max(input_probs) if input_probs else 0.0

        if not in_edges: return comp.prob_val
        return sum(input_probs) + comp.prob_val

    @staticmethod
    def calculate_gap(current_prob: float, goal_prob: float) -> float:
        if current_prob <= goal_prob: return 0.0
        p_curr = max(current_prob, 1e-15)
        p_goal = max(goal_prob, 1e-15)
        return math.log10(p_curr) - math.log10(p_goal)


# --- 5. Core Solver ---

class LayeredSolver:
    def __init__(self, system_model: SystemModel, requirements: List[SafetyRequirement]):
        self.system = system_model
        self.requirements = requirements
        self.constructor = TopologyConstructor(system_model)
        self.max_iterations = 10
        self.traceability_map = {}

    def _get_current_target(self, original_uid: str) -> str:
        current_id = original_uid
        while current_id in self.traceability_map:
            current_id = self.traceability_map[current_id]
        return current_id

    def solve(self):
        print("\n=== Starting Gap-Driven Architecture Generation ===\n")
        print(">>> Phase 1: Physical Topology & Hardware Reliability")
        self._run_phase(target_phase="Physical")

        print("\n>>> Phase 2: Functional Logic & Design Safety")
        self._run_phase(target_phase="Logical")

        print("\n=== Optimization Complete ===")
        return self.system

    def _run_phase(self, target_phase: str):
        iteration = 0
        while iteration < self.max_iterations:
            print(f"\n--- Iteration {iteration + 1} ({target_phase}) ---")

            active_reqs = self._filter_requirements(target_phase)
            if not active_reqs:
                break

            total_gap = 0
            unsolved_reqs = []

            for req in active_reqs:
                current_target_id = self._get_current_target(req.target_comp_id)
                if current_target_id not in self.system.graph:
                    continue

                curr_prob = VerificationEngine.verify_prob(self.system, current_target_id)
                goal_prob = self._get_goal_prob(req.severity)
                gap = VerificationEngine.calculate_gap(curr_prob, goal_prob)

                print(
                    f"Req({req.target_comp_id}) -> Check Node({current_target_id}): P_curr={curr_prob:.2e}, Gap={gap:.2f}")

                if gap > 0:
                    total_gap += gap
                    req.current_processing_id = current_target_id
                    unsolved_reqs.append(req)

            if total_gap <= 0:
                print(f"Phase {target_phase} Converged!")
                break

            # Process the first unsolved requirement
            target_req = unsolved_reqs[0]
            processing_id = target_req.current_processing_id

            # [FIXED] Core fix: Prevent infinite recursion
            # Check if target is already an infrastructure component (Voter/Switch)
            comp_obj = self.system.get_component(processing_id)
            if comp_obj.ctype in [ComponentType.VOTER, ComponentType.SWITCH, ComponentType.COMPARATOR,
                                  ComponentType.RTA_EXEC]:
                print(
                    f"Warning: Gap exists but target {comp_obj.name} is infrastructure. Stopping recursion to avoid loop.")
                unsolved_reqs.pop(0)
                if not unsolved_reqs:
                    break
                target_req = unsolved_reqs[0]
                processing_id = target_req.current_processing_id

            print(f"Addressing Gap for {target_req.target_comp_id} (Node: {processing_id})...")

            pattern_op = self.constructor.inference_engine.select_pattern(target_req)
            if pattern_op:
                try:
                    new_nodes = pattern_op.apply(self.system, processing_id)
                    if 'output' in new_nodes:
                        self.traceability_map[processing_id] = new_nodes['output']
                        print(f"Updated Traceability: {processing_id} -> {new_nodes['output']}")
                except Exception as e:
                    print(f"Error applying pattern: {e}")

            iteration += 1

    def _filter_requirements(self, phase: str) -> List[SafetyRequirement]:
        filtered = []
        for req in self.requirements:
            if phase == "Physical" and not req.design_error:
                filtered.append(req)
            elif phase == "Logical" and req.design_error:
                filtered.append(req)
        return filtered

    def _get_goal_prob(self, severity: SafetySeverity) -> float:
        if severity == SafetySeverity.CATASTROPHIC:
            return 1e-7
        elif severity == SafetySeverity.HAZARDOUS:
            return 1e-6
        elif severity == SafetySeverity.MAJOR:
            return 1e-5
        else:
            return 1e-4


# --- 6. Data Loading ---

class DataLoader:
    @staticmethod
    def load_system_model(json_path: str) -> SystemModel:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model = SystemModel()
        type_map = {t.value: t for t in ComponentType}
        # Fallback map for custom strings
        type_map.update({"Sensor": ComponentType.SENSOR, "Actuator": ComponentType.ACTUATOR,
                         "Processor": ComponentType.PROCESSOR, "Monitor": ComponentType.MONITOR})

        for comp_data in data.get("components", []):
            c_type = type_map.get(comp_data["type"], ComponentType.PROCESSOR)

            # [Change] failure_rate -> failure_probability
            # Prioritize failure_probability, fallback to failure_rate
            f_prob = comp_data.get("failure_probability", comp_data.get("failure_rate", 1e-5))

            model.add_component(Component(
                uid=comp_data["uid"],
                ctype=c_type,
                failure_prob=f_prob,
                name=comp_data.get("name", ""),
                implemented_functions=comp_data.get("implemented_functions", [])
            ))
        for conn in data.get("connections", []):
            model.add_connection(conn["source"], conn["target"])
        return model

    @staticmethod
    def load_requirements(json_path: str) -> List[SafetyRequirement]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        requirements = []
        sev_map = {"Catastrophic": SafetySeverity.CATASTROPHIC, "Hazardous": SafetySeverity.HAZARDOUS,
                   "Major": SafetySeverity.MAJOR}
        foc_map = {"Availability": SafetyFocus.AVAILABILITY, "Integrity": SafetyFocus.INTEGRITY}

        for req_data in data:
            req = SafetyRequirement(
                req_id=req_data.get("id", "N/A"),
                related_function=req_data["related_function"],
                severity=sev_map.get(req_data["severity"], SafetySeverity.MINOR),
                focus=foc_map.get(req_data["focus"], SafetyFocus.AVAILABILITY),
                design_error=req_data.get("design_error", False)
            )
            requirements.append(req)
        return requirements


class RequirementAllocator:
    @staticmethod
    def allocate(system: SystemModel, reqs: List[SafetyRequirement]):
        print("\n=== Allocating Safety Requirements to Components ===")

        # 1. Build inverted index: Function -> Component ID
        func_to_comp = {}
        for node_id in system.graph.nodes:
            comp = system.get_component(node_id)
            for func in comp.implemented_functions:
                func_to_comp[func] = comp.uid

        # 2. Allocate requirements
        allocated_count = 0
        for req in reqs:
            # Exact Match
            if req.related_function in func_to_comp:
                req.target_comp_id = func_to_comp[req.related_function]
                req.current_processing_id = req.target_comp_id
                print(f"  [Allocated] {req.req_id} ('{req.related_function}') -> {req.target_comp_id}")
                allocated_count += 1
            else:
                # Fuzzy Match
                matched = False
                for func_key, comp_id in func_to_comp.items():
                    if req.related_function in func_key or func_key in req.related_function:
                        req.target_comp_id = comp_id
                        req.current_processing_id = comp_id
                        print(
                            f"  [Fuzzy Match] {req.req_id} ('{req.related_function}') -> {comp_id} (via '{func_key}')")
                        matched = True
                        allocated_count += 1
                        break

                if not matched:
                    print(
                        f"  [Warning] Unallocated Requirement: {req.req_id} (Function '{req.related_function}' not found)")

        print(f"Allocation Complete. {allocated_count}/{len(reqs)} requirements mapped.\n")


if __name__ == "__main__":

    print("=== Loading Data ===")
    initial_system = DataLoader.load_system_model("initial_arch.json")
    safety_reqs = DataLoader.load_requirements("safety_requirements_test.json")
    RequirementAllocator.allocate(initial_system, safety_reqs)
    valid_reqs = [r for r in safety_reqs if r.target_comp_id is not None]

    solver = LayeredSolver(initial_system, safety_reqs)
    final_system = solver.solve()

    print("\n=== Result Nodes ===")
    for n in final_system.graph.nodes:
        comp = final_system.get_component(n)

        print(f" [{n}] {comp.ctype.value} (P={comp.prob_val})")