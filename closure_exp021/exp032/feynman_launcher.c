/* Runs <exe dir>\<exe name>.py with aifenv Python; a Job Object kills the child when this launcher is killed on timeout. */
#include <windows.h>
#include <stdio.h>
int main(void){ char me[MAX_PATH], cmd[2*MAX_PATH+200]; GetModuleFileNameA(NULL,me,MAX_PATH); char*d=strrchr(me,'.'); if(d)*d=0;
 snprintf(cmd,sizeof cmd,"\"C:/Users/ianba/Downloads/sr-engines/aifenv/Scripts/python.exe\" \"%s.py\"",me);
 HANDLE j=CreateJobObjectA(NULL,NULL); JOBOBJECT_EXTENDED_LIMIT_INFORMATION i; ZeroMemory(&i,sizeof i); i.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; SetInformationJobObject(j,JobObjectExtendedLimitInformation,&i,sizeof i); AssignProcessToJobObject(j,GetCurrentProcess());
 STARTUPINFOA s; PROCESS_INFORMATION p; ZeroMemory(&s,sizeof s); s.cb=sizeof s; if(!CreateProcessA(NULL,cmd,NULL,NULL,TRUE,0,NULL,NULL,&s,&p)) return 127; WaitForSingleObject(p.hProcess,INFINITE); DWORD c=0; GetExitCodeProcess(p.hProcess,&c); return (int)c; }
