drop table subject_listings;
drop table subjects;
drop table listings;

create table subjects (
    id  serial  primary key
  , name  text  unique  not null
  , date_last_scraped  timestamp  without time zone  not null  default '0001-01-01'
);

create table listings (
    id  bigint  primary key
  , title  text  not null
  , img  text  not null
  , url  text  not null
  , details  text  not null
  , description  text  not null
  , date_posted  timestamp  not null
  , location  text  not null
  , price  money  not null
  , date_scraped  timestamp  without time zone  not null  default (now() at time zone 'utc')
);

create table subject_listings (
    subject_id  int  not null
  , foreign key(subject_id) references subjects(id) on delete cascade
  , listing_id  int  not null
  , foreign key(listing_id) references listings(id) on delete cascade
  , primary key(subject_id, listing_id)
);

-- permissions hack
grant all privileges on table subjects to insight;
grant all privileges on table listings to insight;
grant all privileges on table subject_listings to insight;
grant usage, select on all sequences in schema public to insight;
