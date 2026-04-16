from nicegui import ui, app, events
import warnings
# Filter specific spurious warnings from pyannote/torchcodec on Windows
warnings.filterwarnings("ignore", module="pyannote.audio.core.io", message="torchcodec is not installed correctly")

import os
import asyncio
from audio_processor import AudioPipeline
import matplotlib
# Force non-interactive backend to prevent Tkinter threading crashes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from dataclasses import dataclass, field
from typing import List, Optional

# --- Data Structures ---
@dataclass
class ProcessedFile:
    id: str
    path: str
    name: str
    status: str = "Pending" # Pending, Processing, Done, Error
    vocals_file: Optional[str] = None
    music_file: Optional[str] = None
    segments: List[dict] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

# --- Global State ---
class AppState:
    def __init__(self):
        self.hf_token = ""
        self.queue: List[ProcessedFile] = []
        self.current_file_id = None # ID of file currently being viewed/edited
        self.is_processing = False
        self.trim_duration = 30.0
        self.pipeline = None

state = AppState()

# Ensure directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- UI Components ---

def processing_ui():
    ui.add_head_html('<style>body { background-color: #f0f4f8; }</style>')

    with ui.column().classes('w-full items-center max-w-6xl mx-auto p-4'):
        ui.markdown("# 🎙️ Podcast Multi-Speaker Batch Editor").classes('text-3xl font-bold text-gray-800')

        # 1. Setup & Upload
        with ui.card().classes('w-full p-4 mb-4'):
            ui.markdown("## 1. Setup & Upload")
            with ui.row().classes('w-full gap-4'):
                ui.input(
                    label="Hugging Face Token", 
                    password=True, 
                    on_change=lambda e: setattr(state, 'hf_token', e.value)
                ).bind_value(state, 'hf_token').classes('flex-grow').props('outlined')

                ui.number(
                    label="Trim Start (seconds)", 
                    value=30.0,
                    min=0.0,
                    on_change=lambda e: setattr(state, 'trim_duration', e.value)
                ).bind_value(state, 'trim_duration').classes('w-32').props('outlined')

            ui.upload(
                label="Upload Podcasts (MP3/WAV)", 
                multiple=True,
                auto_upload=True, 
                on_upload=handle_upload
            ).classes('w-full mt-4').props('accept=.mp3,.wav')

        # 2. Queue & Processing
        with ui.card().classes('w-full p-4 mb-4').bind_visibility_from(state, 'queue', backward=lambda x: len(x) > 0):
            with ui.row().classes('w-full items-center justify-between'):
                ui.markdown("## 2. Processing Queue")
                ui.button("Start Batch Processing", on_click=start_batch_processing).props('color=primary').bind_enabled_from(state, 'is_processing', backward=lambda x: not x)

            # Queue Table
            def get_queue_data():
                return [{'Name': f.name, 'Status': f.status, 'ID': f.id} for f in state.queue]

            queue_grid = ui.aggrid({
                'columnDefs': [
                    {'headerName': 'Name', 'field': 'Name', 'flex': 1},
                    {'headerName': 'Status', 'field': 'Status', 'width': 120},
                ],
                'rowData': get_queue_data(),
            }).classes('h-48')

            def refresh_queue_ui():
                queue_grid.options['rowData'] = get_queue_data()
                queue_grid.update()

            ui.timer(1.0, refresh_queue_ui)

        # 3. Editor
        with ui.column().classes('w-full').bind_visibility_from(state, 'queue', backward=lambda q: any(f.status == "Done" for f in q)):
            with ui.card().classes('w-full p-4 mb-4'):
                ui.markdown("## 3. Review & Edit")

                # File Selector
                processed_files = lambda: [f for f in state.queue if f.status == "Done"]

                select = ui.select(
                    options={}, 
                    label="Select File to Edit", 
                    on_change=lambda e: load_editor_for_file(e.value)
                ).classes('w-full mb-4')

                def update_select_options():
                    opts = {f.id: f.name for f in processed_files()}
                    select.options = opts
                    select.update()
                    if not select.value and opts:
                        select.value = list(opts.keys())[0]
                        load_editor_for_file(select.value)

                ui.timer(1.0, update_select_options)

                editor_container = ui.column().classes('w-full')

def load_editor_for_file(file_id):
    if not file_id: return
    state.current_file_id = file_id

    file_obj = next((f for f in state.queue if f.id == file_id), None)
    if not file_obj: return

    render_editor(file_obj)

@ui.refreshable
def render_editor(file_obj: ProcessedFile):
    ui.markdown(f"### Editing: {file_obj.name}")

    # Waveform
    with ui.card().classes('w-full mb-2'):
        plot_container = ui.element('div').classes('w-full')
        with plot_container:
            create_waveform_plot(file_obj)

    # Segments Grid
    ui.markdown("#### Segments")
    grid = ui.aggrid({
        'columnDefs': [
            {'headerName': 'Start (s)', 'field': 'start', 'sortable': True, 'width': 100},
            {'headerName': 'End (s)', 'field': 'end', 'width': 100},
            {'headerName': 'Duration', 'valueGetter': 'data.end - data.start', 'width': 100},
            {'headerName': 'Speaker', 'field': 'label', 'editable': True, 'cellEditor': 'agSelectCellEditor', 
                'cellEditorParams': {'values': ['SPEAKER_00', 'SPEAKER_01', 'Overlap']}}
        ],
        'rowData': file_obj.segments,
        'stopEditingWhenCellsLoseFocus': True,
    }).classes('h-96')

    async def export_tracks_current():
        updated_data = await grid.get_row_data()
        file_obj.segments = updated_data

        ui.notify(f"Exporting {file_obj.name}...")
        await asyncio.to_thread(run_export, file_obj)
        ui.notify("Export Complete! Check 'output' folder.")

    ui.button("Export Corrected Tracks", on_click=export_tracks_current).props('color=secondary icon=download')

import uuid

async def handle_upload(e: events.UploadEventArguments):
    try:
        filename = getattr(e, 'name', None)
        if not filename:
             if hasattr(e, 'file'):
                 filename = getattr(e.file, 'filename', None) or getattr(e.file, 'name', None)

        if not filename:
             filename = "unknown_file_" + str(uuid.uuid4())[:4]

        filename = os.path.basename(filename)
        file_path = os.path.join(UPLOAD_DIR, filename)

        if hasattr(e, 'content'):
            with open(file_path, 'wb') as f:
                content = e.content.read()
                if asyncio.iscoroutine(content):
                    content = await content
                f.write(content)
        elif hasattr(e, 'file'):
            with open(file_path, 'wb') as f:
                content_source = getattr(e.file, 'file', e.file)
                if hasattr(content_source, 'seek'):
                    content_source.seek(0)

                content = content_source.read()
                if asyncio.iscoroutine(content):
                    content = await content
                f.write(content)

        new_id = str(uuid.uuid4())[:8]
        new_file = ProcessedFile(id=new_id, path=file_path, name=filename)
        state.queue.append(new_file)

        ui.notify(f"Added to queue: {filename}")
    except Exception as err:
        ui.notify(f"Upload failed: {err}", type='negative')
        print(f"Upload Error: {err}")

async def start_batch_processing():
    if not state.hf_token:
        ui.notify("Please enter Hugging Face Token first!", type='negative')
        return

    state.is_processing = True
    files_to_process = [f for f in state.queue if f.status == "Pending"]

    if not files_to_process:
        ui.notify("No pending files.")
        state.is_processing = False
        return

    ui.notify(f"Starting batch of {len(files_to_process)} files...")

    for file_obj in files_to_process:
        file_obj.status = "Processing"
        try:
            await asyncio.to_thread(process_single_file, file_obj)
            file_obj.status = "Done"
            ui.notify(f"Finished: {file_obj.name}")
        except Exception as e:
            file_obj.status = "Error"
            file_obj.log.append(str(e))
            ui.notify(f"Error processing {file_obj.name}: {e}", type='negative')
            print(f"Error processing {file_obj.name}: {e}")

    state.is_processing = False
    ui.notify("Batch Processing Complete!")

def process_single_file(file_obj: ProcessedFile):
    pipeline = AudioPipeline(hf_token=state.hf_token)
    v_path, m_path = pipeline.separate_music(file_obj.path, output_dir=OUTPUT_DIR)
    file_obj.vocals_file = v_path
    file_obj.music_file = m_path

    diarization = pipeline.diarize(v_path)
    segments = pipeline.process_segments(diarization)
    file_obj.segments = segments

    export_dir = os.path.join(OUTPUT_DIR, "final_" + file_obj.name)
    pipeline.export_audio(segments, v_path, output_dir=export_dir, trim_start_sec=state.trim_duration)

def run_export(file_obj: ProcessedFile):
    pipeline = AudioPipeline(hf_token=state.hf_token)
    export_dir = os.path.join(OUTPUT_DIR, "exported_" + file_obj.name)
    pipeline.export_audio(file_obj.segments, file_obj.vocals_file, output_dir=export_dir, trim_start_sec=state.trim_duration)

def create_waveform_plot(file_obj):
    if not file_obj.vocals_file: return

    try:
        audio = AudioSegment.from_file(file_obj.vocals_file)
        audio = audio.set_frame_rate(1000)
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2: samples = samples[::2]
        times = np.linspace(0, len(audio)/1000, num=len(samples))

        with ui.pyplot(figsize=(10, 3)):
            plt.fill_between(times, samples, color='lightgray', label='Audio')
            colors = {'SPEAKER_00': 'blue', 'SPEAKER_01': 'green', 'Overlap': 'red'}
            for seg in file_obj.segments:
                start = seg['start']
                end = seg['end']
                label = seg['label']
                c = colors.get(label, 'gray')
                plt.axvspan(start, end, alpha=0.3, color=c, ymin=0, ymax=1)
            plt.xlabel("Time (s)")
            plt.title("Speaker Segmentation")
    except Exception as e:
        ui.label(f"Plot Error: {e}")

if __name__ in {"__main__", "__mp_main__"}:
    processing_ui()
    ui.run(title="Podcast Voice Extractor", show=False, reload=False)
