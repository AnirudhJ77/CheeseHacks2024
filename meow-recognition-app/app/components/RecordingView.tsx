"use client"; // This is a client component 
import { useState, useRef, useEffect } from "react";
import { WaveFile } from "wavefile"; // npm install wavefile


export default function RecordingView() {
    const [isRecording, setIsRecording] = useState<boolean>(false);
    const [recordingComplete, setRecordingComplete] = useState<boolean>(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [file, setFile] = useState<Blob | null>(null);
    const [hasResult, setHasResult] = useState<boolean>(false);
    const [description, setDescription] = useState<string>("Your cat is hungry");


    //#region Audio Recording Logic
    // Convert Float32Array to Int16Array
    const float32ToInt16 = (buffer: Float32Array): Int16Array => {
        const int16Buffer = new Int16Array(buffer.length);
        for (let i = 0; i < buffer.length; i++) {
            int16Buffer[i] = Math.max(-1, Math.min(1, buffer[i])) * 0x7FFF; // Clamp to 16-bit range
        }
        return int16Buffer;
    };

    const downsampleBuffer = (buffer: Float32Array, originalRate: number, targetRate: number): Int16Array => {
        if (originalRate === targetRate) {
            return float32ToInt16(buffer); // No resampling needed
        }
        const sampleRatio = originalRate / targetRate;
        const newLength = Math.round(buffer.length / sampleRatio);
        const downsampledBuffer = new Int16Array(newLength);
        for (let i = 0; i < newLength; i++) {
            downsampledBuffer[i] = Math.max(-1, Math.min(1, buffer[Math.round(i * sampleRatio)])) * 0x7FFF; // Clamp and convert to 16-bit
        }
        return downsampledBuffer;
    };


    const processToWav = async (blob: Blob): Promise<Uint8Array> => {
        // Decode WebM/Opus audio using AudioContext
        const audioContext = new AudioContext();
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // Extract PCM data from the audio buffer
        const sampleRate = 48000; // (adjust as needed)
        const channelData = audioBuffer.getChannelData(0); // Get data from the first channel
        const pcmData = downsampleBuffer(channelData, audioBuffer.sampleRate, sampleRate);

        // Encode PCM data into WAV
        const wav = new WaveFile();
        wav.fromScratch(1, sampleRate, "16", pcmData); // Mono, sample rate, 16-bit
        return wav.toBuffer();
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: "audio/webm;codecs=opus",
            });
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
                const wavBuffer = await processToWav(blob);
                const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });

                // Send the file to Flask backend
                const formData = new FormData();
                formData.append("wavFile", wavBlob, "audio.wav");

                try {
                    const response = await fetch("http://34.68.139.216:5000/", {
                        method: "POST",
                        body: formData,
                    });
                    console.log(response.body);
                    if (response.ok) {
                        const result = await response.json();
                        console.log("Analysis result:", result);
                        setDescription(result.analysis); // Update your UI with Flask response
                    } else {
                        console.error("Failed to upload WAV file");
                    }
                } catch (error) {
                    console.error("Error uploading WAV file:", error);
                }

                const url = URL.createObjectURL(wavBlob);
                setAudioUrl(url);
                setRecordingComplete(true);

                setHasResult(true);
            };

            mediaRecorder.start();
            setIsRecording(true);
            setHasResult(false);

        } catch (error) {
            console.error("Error accessing microphone:", error);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const handleToggleRecording = () => {
        setIsRecording(!isRecording)
        if (!isRecording) {
            startRecording()
        }
        else stopRecording();
    }

    //#endregion

    return (
        <div className="flex items-center justify-center h-screen w-full">
            <div className="">
                <h1 className="text-center text-9xl font-mono font-bold text-amber-800 opacity-70">
                    ShaMeow
                    <p className="text-center text-xl font-mono font-bold text-amber-800 opacity-70">
                        Find out what your cat is thinking!
                    </p>
                    <p className="text-center text-s font-mono font-bold text-amber-800 opacity-60">
                    </p>
                </h1>
                {/* Button Section */}
                <div className="flex items-center w-full">
                    {isRecording ? (
                        <button onClick={handleToggleRecording}
                            className="rounded-full w-40 h-40 mt-10 m-auto flex items-center justify-center bg-red-400 hover:bg-red-500 z-10"
                        >
                            <svg
                                className="w-24 h-24 z-10"
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 32 32"><path fill="#ffffff"
                                    d="M12 6h-2a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2zm10 0h-2a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2z" />
                            </svg>
                            <span className="animate-ping absolute inline-flex h-40 w-40 rounded-full bg-red-300 opacity-75 z-0"></span>
                        </button>

                    ) : (
                        <button onClick={handleToggleRecording}
                            className="animate-pulse rounded-full w-40 h-40 mt-10 m-auto flex items-center justify-center bg-amber-500 opacity-80 hover:bg-amber-600"
                        >
                            <svg
                                className="w-24 h-24"
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 24 24">
                                <g fill="none" stroke="#ffffff" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5">
                                    <path d="M15.64 5.836c0-1.704-1.63-3.086-3.64-3.086c-2.01 0-3.64 1.382-3.64 3.086v5.575c0 1.704 1.63 3.086 3.64 3.086c2.01 0 3.64-1.382 3.64-3.086z" />
                                    <path d="M5.328 10.616a6.672 6.672 0 1 0 13.344 0M12 21.25v-3.962M8.36 21.25h7.28" />
                                </g>
                            </svg>
                        </button>
                    )}
                </div>
                {/*
                <div className="pt-5 pb-5">
                    {recordingComplete && audioUrl && (
                        <div className="flex items-center justify-center w-full">
                            <audio controls src={audioUrl}></audio>
                            <a className="flex items-center justify-center pl-5"
                                href={audioUrl}
                                download="cat-audio.wav">
                                Download WAV
                            </a>
                        </div>
                    )}
                </div>
                */}

                <div className="flex items-center justify-center">
                    <div className={`transition-opacity duration-1000 ${hasResult ? "opacity-100" : "opacity-0"} 
                    bg-opacity-20 w-full md:w-[800px] h-20 p-4 text-balance text-center text-3xl font-mono font-bold text-amber-800`}>
                        <p className=" bg-opacity-20 w-full md:w-[800px] h-20 p-4 text-balance text-center text-3xl font-mono font-bold text-amber-800 opacity-50">
                            {description}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );

}