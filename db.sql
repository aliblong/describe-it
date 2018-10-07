drop database if exists kijiji;
drop role if exists descrive;

create database kijiji;
create role descrive with password 'descrive';
grant all privileges on database kijiji to descrive;
ALTER ROLE descrive WITH LOGIN;
