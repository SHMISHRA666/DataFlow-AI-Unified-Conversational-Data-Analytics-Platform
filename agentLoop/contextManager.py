# contextManager.py - Essential Functionality + Simple Output Chain (no networkx)

import json
import time
from datetime import datetime
from pathlib import Path
import asyncio
from action.executor import run_user_code
from utils.utils import log_step, log_error
import pdb

class ExecutionContextManager:
    def __init__(self, plan_graph: dict, session_id: str = None, original_query: str = None, file_manifest: list = None, debug_mode: bool = False):
        # Minimal graph-like container
        self.plan_graph = {
            'graph': {
                'session_id': session_id or str(int(time.time()))[-8:],
                'original_query': original_query,
                'file_manifest': file_manifest or [],
                'created_at': datetime.utcnow().isoformat(),
                'status': 'running',
                'output_chain': {},
                'validation_results': {"is_valid": True, "errors": [], "warnings": []}
            },
            'nodes': {},   # id -> node data
            'edges': []    # list of (source, target)
        }
        self.debug_mode = debug_mode

        # Add ROOT node
        self.plan_graph['nodes']["ROOT"] = {
            'id': 'ROOT', 'description': 'Initial Query', 'agent': 'System',
            'status': 'completed', 'output': None, 'error': None, 'cost': 0.0,
            'start_time': None, 'end_time': None, 'execution_time': 0.0,
            'reads': [], 'writes': []
        }

        # Add nodes
        for node in plan_graph.get("nodes", []):
            self.plan_graph['nodes'][node['id']] = {
                **node,
                'status': 'pending', 'output': None, 'error': None, 'cost': 0.0,
                'start_time': None, 'end_time': None, 'execution_time': 0.0
            }
        # Add edges
        for edge in plan_graph.get("edges", []):
            self.plan_graph['edges'].append((edge['source'], edge['target']))

    def _node(self, node_id: str) -> dict:
        return self.plan_graph['nodes'][node_id]

    def _predecessors(self, node_id: str):
        return [s for (s, t) in self.plan_graph['edges'] if t == node_id]

    def get_ready_steps(self):
        # Not used in sequential flow; keep simple readiness check
        ready = []
        for nid, data in self.plan_graph['nodes'].items():
            if nid == 'ROOT':
                continue
            if data['status'] != 'pending':
                continue
            preds = self._predecessors(nid)
            if all(self._node(p)['status'] == 'completed' for p in preds):
                ready.append(nid)
        return ready

    def get_inputs(self, reads):
        inputs = {}
        output_chain = self.plan_graph['graph']['output_chain']
        for step_id in reads or []:
            if step_id in output_chain:
                inputs[step_id] = output_chain[step_id]
            else:
                log_step(f"⚠️  Missing dependency: '{step_id}' not found", symbol="❓")
        return inputs

    def mark_running(self, step_id):
        node = self._node(step_id)
        node['status'] = 'running'
        node['start_time'] = datetime.utcnow().isoformat()
        self._auto_save()

    def _has_executable_code(self, output):
        if not isinstance(output, dict):
            return False
        return ("files" in output or "code_variants" in output or 
                any(k.startswith("CODE_") for k in output.keys()) or
                any(key in output for key in ["tool_calls", "schedule_tool", "browser_commands", "python_code"]))

    async def _auto_execute_code(self, step_id, output):
        node_data = self._node(step_id)
        reads = node_data.get("reads", [])
        reads_data = {}
        output_chain = self.plan_graph['graph']['output_chain']
        for read_key in reads:
            if read_key in output_chain:
                reads_data[read_key] = output_chain[read_key]
        try:
            result = await run_user_code(
                output_data=output,
                multi_mcp=getattr(self, 'multi_mcp', None),
                session_id=self.plan_graph['graph']['session_id'],
                inputs=reads_data
            )
            node_data['execution_result'] = result
            return result
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            log_error(error_msg)
            return {"status": "failed", "error": error_msg}

    async def mark_done(self, step_id, output=None, cost=None, input_tokens=None, output_tokens=None):
        final_output = output
        execution_result = None
        if output and self._has_executable_code(output):
            log_step(f"🔧 Executing code for {step_id}", symbol="⚙️")
            execution_result = await self._auto_execute_code(step_id, output)
            if isinstance(output, dict) and execution_result.get("status") == "success":
                final_output = output.copy()
                final_output["execution_result"] = execution_result
                if execution_result.get("result"):
                    if isinstance(execution_result["result"], dict):
                        final_output.update(execution_result["result"])
                    else:
                        final_output["data"] = execution_result["result"]
                if execution_result.get("created_files"):
                    final_output["created_files"] = execution_result["created_files"]

        self.plan_graph['graph']['output_chain'][step_id] = final_output

        node = self._node(step_id)
        node.update({
            'status': 'completed',
            'output': final_output,
            'cost': cost or 0.0,
            'input_tokens': input_tokens or 0,
            'output_tokens': output_tokens or 0,
            'end_time': datetime.utcnow().isoformat(),
            'execution_result': execution_result
        })
        if node['start_time']:
            start = datetime.fromisoformat(node['start_time'])
            end = datetime.fromisoformat(node['end_time'])
            node['execution_time'] = (end - start).total_seconds()
        log_step(f"✅ {step_id} completed - output stored in chain", symbol="📦")
        self._auto_save()

    def mark_failed(self, step_id, error=None):
        node = self._node(step_id)
        node.update({
            'status': 'failed',
            'end_time': datetime.utcnow().isoformat(),
            'error': str(error) if error else None
        })
        if node['start_time']:
            start = datetime.fromisoformat(node['start_time'])
            end = datetime.fromisoformat(node['end_time'])
            node['execution_time'] = (end - start).total_seconds()
        log_error(f"❌ {step_id} failed: {error}")
        self._auto_save()

    def get_step_data(self, step_id):
        return self._node(step_id)

    def all_done(self):
        return all(data['status'] in ['completed', 'failed']
                   for nid, data in self.plan_graph['nodes'].items()
                   if nid != 'ROOT')

    def get_execution_summary(self):
        completed = sum(1 for nid, data in self.plan_graph['nodes'].items()
                        if nid != 'ROOT' and data.get('status') == 'completed')
        failed = sum(1 for nid, data in self.plan_graph['nodes'].items()
                     if nid != 'ROOT' and data.get('status') == 'failed')
        total = len(self.plan_graph['nodes']) - 1
        total_cost = sum(self.plan_graph['nodes'][nid].get('cost', 0.0)
                         for nid in self.plan_graph['nodes'] if nid != 'ROOT')
        total_input_tokens = sum(self.plan_graph['nodes'][nid].get('input_tokens', 0)
                                 for nid in self.plan_graph['nodes'] if nid != 'ROOT')
        total_output_tokens = sum(self.plan_graph['nodes'][nid].get('output_tokens', 0)
                                  for nid in self.plan_graph['nodes'] if nid != 'ROOT')
        return {
            'session_id': self.plan_graph['graph']['session_id'],
            'original_query': self.plan_graph['graph']['original_query'],
            'completed_steps': completed,
            'failed_steps': failed,
            'total_steps': total,
            'total_cost': total_cost,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'output_chain': self.plan_graph['graph']['output_chain']
        }

    def set_multi_mcp(self, multi_mcp):
        self.multi_mcp = multi_mcp

    def _auto_save(self):
        # Disabled (no serializer)
        return

    def get_session_data(self):
        return {
            'session_id': self.plan_graph['graph']['session_id'],
            'output_chain': self.plan_graph['graph']['output_chain'],
            'nodes': self.plan_graph['nodes'],
            'links': self.plan_graph['edges'],
            'original_query': self.plan_graph['graph'].get('original_query', ''),
            'created_at': self.plan_graph['graph'].get('created_at', ''),
            'execution_summary': self.get_execution_summary()
        }

    @classmethod
    def load_session(cls, session_file: Path, debug_mode: bool = False):
        context = cls.__new__(cls)
        context.plan_graph = {'graph': {}, 'nodes': {}, 'edges': []}
        context.debug_mode = debug_mode
        return context
