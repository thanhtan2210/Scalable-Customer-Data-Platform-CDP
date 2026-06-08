import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../api/client';
import { Upload as UploadIcon, Loader2 } from 'lucide-react';

const Upload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const { data } = await apiClient.post('/datasets/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      // Auto-trigger profiling
      await apiClient.post(`/datasets/${data.dataset_id}/profile`);
      navigate(`/review/${data.dataset_id}`);
    } catch (error) {
      console.error('Upload failed', error);
      alert('Upload failed. Please check the file format and size.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto mt-20 p-8 border-2 border-dashed border-slate-300 rounded-xl bg-white shadow-sm">
      <div className="flex flex-col items-center justify-center space-y-4">
        <div className="p-4 bg-blue-50 rounded-full">
          <UploadIcon className="w-12 h-12 text-blue-500" />
        </div>
        <h2 className="text-2xl font-bold text-slate-800">Upload Dataset</h2>
        <p className="text-slate-500 text-center">
          Kéo thả file CSV hoặc Excel của bạn vào đây (Tối đa 50MB)
        </p>
        
        <input 
          type="file" 
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />

        {file && (
          <div className="w-full p-4 bg-slate-50 rounded-lg flex justify-between items-center">
            <span className="font-medium truncate">{file.name}</span>
            <span className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="w-full py-3 px-6 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 disabled:bg-slate-300 transition-all flex justify-center items-center"
        >
          {loading ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Đang phân tích cấu trúc dữ liệu...
            </>
          ) : 'Tiếp tục'}
        </button>
      </div>
    </div>
  );
};

export default Upload;
