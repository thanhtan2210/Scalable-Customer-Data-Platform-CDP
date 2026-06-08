import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import apiClient from '../api/client';
import { AlertCircle, CheckCircle2, Play } from 'lucide-react';

interface ColumnProfile {
  name: str;
  inferred_role: string;
  transform_strategy: string;
  confidence: number;
}

const ColumnReview: React.FC = () => {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const [profiles, setProfiles] = useState<ColumnProfile[]>([]);
  const [target, setTarget] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProfiles = async () => {
      const { data } = await apiClient.get(`/datasets/${datasetId}/profile_result`); // Simplified for MVP
      setProfiles(data.profiles);
      setTarget(data.suggested_target);
      setLoading(false);
    };
    fetchProfiles();
  }, [datasetId]);

  const startTraining = async () => {
    const { data } = await apiClient.post(`/jobs/datasets/${datasetId}/train`, {
      confirmed_target: target,
      confirmed_profiles: profiles
    });
    navigate(`/jobs/${data.job_id}`);
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div className="container mx-auto p-8">
      <header className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Review Data Schema</h1>
          <p className="text-slate-500">Kiểm tra và điều chỉnh các cột trước khi huấn luyện mô hình</p>
        </div>
        <button 
          onClick={startTraining}
          className="flex items-center px-6 py-3 bg-green-600 text-white rounded-lg font-bold hover:bg-green-700 shadow-lg"
        >
          <Play className="mr-2 w-5 h-5" /> Bắt đầu Training
        </button>
      </header>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <table className="w-full text-left">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <th className="px-6 py-4 font-semibold text-slate-700">Column Name</th>
              <th className="px-6 py-4 font-semibold text-slate-700">Role</th>
              <th className="px-6 py-4 font-semibold text-slate-700">Transform</th>
              <th className="px-6 py-4 font-semibold text-slate-700">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {profiles.map((p) => (
              <tr key={p.name} className={p.name === target ? 'bg-blue-50/50' : ''}>
                <td className="px-6 py-4 font-medium flex items-center">
                  {p.name}
                  {p.name === target && <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded uppercase tracking-wider font-bold">Target</span>}
                </td>
                <td className="px-6 py-4">
                  <select 
                    value={p.inferred_role}
                    onChange={(e) => {/* logic to update profile */}}
                    className="bg-transparent border-none focus:ring-0 text-slate-600 cursor-pointer"
                  >
                    <option value="numeric">Numeric</option>
                    <option value="categorical">Categorical</option>
                    <option value="target">Target</option>
                    <option value="id">ID</option>
                    <option value="drop">Drop</option>
                  </select>
                </td>
                <td className="px-6 py-4 text-slate-500 font-mono text-sm">{p.transform_strategy}</td>
                <td className="px-6 py-4">
                  <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                    p.confidence > 0.8 ? 'bg-green-100 text-green-700' : 
                    p.confidence > 0.5 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
                  }`}>
                    {(p.confidence * 100).toFixed(0)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ColumnReview;
