import React from 'react';
import {BrowserRouter,Routes,Route} from 'react-router-dom';
import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import LandingPage from '@/pages/LandingPage';
import VisionExplorerPage from '@/pages/VisionExplorerPage';
import LibraryPage from '@/pages/LibraryPage';
import SearchPage from '@/pages/SearchPage';
import AnalyticsPage from '@/pages/AnalyticsPage';
import ExportPage from '@/pages/ExportPage';
import NotFoundPage from '@/pages/NotFoundPage';
import {StudioLayout} from '@/components/layout/StudioLayout';
import {UploadDialog} from '@/features/uploads/UploadDialog';
import {ProcessingDialog} from '@/features/uploads/ProcessingDialog';
import {useSessionRestore} from '@/features/session/useSessionRestore';
import {AppErrorBoundary} from '@/components/shared/AppErrorBoundary';
const queryClient=new QueryClient({defaultOptions:{queries:{staleTime:30000,retry:1,refetchOnWindowFocus:false}}});
function SessionRestore(){useSessionRestore();return null;}
export default function App(){return <AppErrorBoundary><QueryClientProvider client={queryClient}><BrowserRouter><SessionRestore/><Routes><Route path="/" element={<LandingPage/>}/><Route element={<StudioLayout/>}><Route path="studio" element={<VisionExplorerPage/>}/><Route path="library" element={<LibraryPage/>}/><Route path="search" element={<SearchPage/>}/><Route path="analytics" element={<AnalyticsPage/>}/><Route path="export" element={<ExportPage/>}/></Route><Route path="*" element={<NotFoundPage/>}/></Routes><UploadDialog/><ProcessingDialog/></BrowserRouter></QueryClientProvider></AppErrorBoundary>}
