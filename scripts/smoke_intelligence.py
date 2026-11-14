"""Optional live Gemini smoke test using synthetic source text only (not an accuracy evaluation)."""
import json
from dataclasses import asdict
from pathlib import Path
from exam_prep.agent import Services, build_graph, run_agent
from exam_prep.intelligence import generate_practice, grade_answer
from exam_prep.llm import GeminiLLM
from exam_prep.retrieval import Chunk, MemoryBackend, Retriever

chunk = Chunk(chunk_id='SYN-AI_2025_01_MID-Q1a', paper_id='SYN-AI_2025_01_MID', label='Q1a',
              subject='AI', source='SYNTHETIC', year=2025, exam_type='mid', marks=4,
              topic_ids=['SYN-AI-U1-T1'], topics=['Admissible heuristics'],
              text='Explain what makes a heuristic admissible in A-star search.',
              index_text='Explain what makes a heuristic admissible in A-star search.')
retriever = Retriever(MemoryBackend([chunk], None), None)
llm = GeminiLLM()
practice = generate_practice('admissible heuristic', retriever, llm, 'AI', marks=4)
grade = grade_answer(practice.question, practice.rubric,
                     'An admissible heuristic never overestimates the true remaining cost. '
                     'A-star ranks nodes by f(n)=g(n)+h(n), combining cost so far and estimated remaining cost.', llm)
graph = run_agent('search_papers', {'query':'admissible heuristic','subject':'AI','mode':'bm25'},
                  Services(retriever_for=lambda mode:retriever,chunks=lambda:[chunk],llm=lambda:llm), build_graph())
record = {'kind':'synthetic_live_smoke_not_accuracy_evaluation', 'practice_prompt':practice.prompt_version,
          'question':practice.question, 'marks':practice.marks,
          'rubric':[p.model_dump() for p in practice.rubric], 'grade':asdict(grade),
          'graph_search_hits':len(graph['result']['hits'])}
destination = Path(__file__).resolve().parents[2] / 'output/intelligence_smoke.json'
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'practice_validated':True,'grading_bounded':0<=grade.total<=grade.max_total,
                  'human_validated':False,'graph_hits':record['graph_search_hits']}))
