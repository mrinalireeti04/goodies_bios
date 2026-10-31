import {Link} from 'react-router-dom';
import {Button} from '@/components/ui/button';
export default function NotFoundPage(){return <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-slate-50 text-slate-800"><h1 className="text-5xl font-extrabold">404</h1><p>This page doesn't exist.</p><Button asChild><Link to="/">Return home</Link></Button></div>}
