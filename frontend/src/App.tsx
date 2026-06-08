import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Activity, Upload as UploadIcon, Database, BarChart3, ShieldCheck } from 'lucide-react';
import Upload from './pages/Upload';
import ColumnReview from './pages/ColumnReview';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="min-h-screen bg-slate-50 flex">
    {/* Sidebar */}
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col">
      <div className="p-6 border-b border-slate-200">
        <div className="flex items-center space-x-2 text-blue-600">
          <ShieldCheck className="w-8 h-8" />
          <span className="text-xl font-black tracking-tighter text-slate-900 italic">CHURN.AI</span>
        </div>
      </div>
      <nav className="flex-1 p-4 space-y-1">
        <Link to="/upload" className="flex items-center px-4 py-2 text-slate-600 hover:bg-blue-50 hover:text-blue-600 rounded-lg transition-colors">
          <UploadIcon className="w-5 h-5 mr-3" /> Upload
        </Link>
        <Link to="/models" className="flex items-center px-4 py-2 text-slate-600 hover:bg-blue-50 hover:text-blue-600 rounded-lg transition-colors">
          <Database className="w-5 h-5 mr-3" /> Model Hub
        </Link>
        <Link to="/dashboard" className="flex items-center px-4 py-2 text-slate-600 hover:bg-blue-50 hover:text-blue-600 rounded-lg transition-colors">
          <BarChart3 className="w-5 h-5 mr-3" /> Dashboard
        </Link>
      </nav>
      <div className="p-4 border-t border-slate-100">
        <div className="bg-blue-600 p-4 rounded-xl text-white">
          <p className="text-xs font-medium opacity-80">System Status</p>
          <div className="flex items-center mt-1">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-2" />
            <span className="text-sm font-bold">API Healthy</span>
          </div>
        </div>
      </div>
    </aside>

    {/* Main Content */}
    <main className="flex-1 overflow-auto">
      {children}
    </main>
  </div>
);

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/upload" element={<Upload />} />
          <Route path="/review/:datasetId" element={<ColumnReview />} />
          <Route path="/" element={<Upload />} />
          {/* Add other routes as needed */}
        </Routes>
      </Layout>
    </BrowserRouter>
  );
};

export default App;
